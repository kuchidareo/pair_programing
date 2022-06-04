from transformers import AutoTokenizer, AutoModel, AutoModelForMultipleChoice
from datasets import Features, load_dataset, ClassLabel, Value
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import time

model_name = "tohoku_bert_manual"
import_model = "rinna/japanese-roberta-base"

MAX_LENGTH = 80
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
data_dir = {"train": "./KUCI/train.jsonl", "development": "./KUCI/development.jsonl", "test": "./KUCI/test.jsonl"}
dataset_features=Features({
        "id": Value("int64"),
        "label": ClassLabel(names=["a", "b", "c", "d"]),
        "agreement": Value("int64"),
        "context": Value("string"),
        "choice_a": Value("string"),
        "choice_b": Value("string"),
        "choice_c": Value("string"),
        "choice_d": Value("string"),
})

class MultipleChoiceModel(nn.Module):
    def __init__(self):
        super(MultipleChoiceModel, self).__init__()
        self.bert = AutoModel.from_pretrained(import_model)
        self.bert = nn.DataParallel(self.bert)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(768, 1)

    def forward(self, x):
        src = []
        for q in x:
            t1 = self.bert(input_ids = q[0], attention_mask = q[1], token_type_ids = q[2])
            t2 = self.dropout(t1.pooler_output)
            t3 = self.linear(t2).squeeze()
            src.append(t3)

        logits = torch.t(torch.stack(src, dim=0)) # torch.Size([batch, 4])
        return logits
            

"""
pre-processing of context and choice sentences
input: huggingface_datasets
output:
{"input_ids": [
                [tensor0, tensor1, tensor2, tensor3] - Q1,
                [tensor0, tensor1, tensor2, tensor3] - Q2, ...
              ]      
 "token_type_ids": [
                     [tensor0, ... , tensor3],...
                   ]
 "attention_mask: [
                     [tensor0, ..., tensor3],...
       
                  ] 
}
"""
def data_to_tensor_features(data):
    choices = ["choice_a", "choice_b", "choice_c", "choice_d"]
    context_sentences_list = [["[CLS]" + " " + context] * 4 for context in data["context"]]
    choice_sentences_list = [["[SEP]" + " " + data[choice][i] + " " + "[SEP]" for choice in choices] for i in range(len(data["context"]))]

    context_sentences_list = sum(context_sentences_list, [])
    choice_sentences_list = sum(choice_sentences_list, [])

    tokenized_connected_sentences = tokenizer(
        context_sentences_list,
        choice_sentences_list,
        padding = "max_length", 
        max_length = MAX_LENGTH, 
        truncation = True,
        add_special_tokens=False
    )

    # k = <str> e.g, "input_ids"
    # v = <Tensor> len(v) = 4 * len(context)
    features = {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_connected_sentences.items()}
    
    return features


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


dataset = load_dataset("json", data_files={"train": data_dir["train"], "development": data_dir["development"]}, features=dataset_features)
encoded_dataset = dataset.map(data_to_tensor_features, batched=True)

encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])

# ここで順番が入れ替わる
# data_size - 4 => 4 - data_size
# 勝手に階層構造を変更されるので、train時に変更し直す。行列でいう転置をする。
train_dataloader = DataLoader(encoded_dataset["train"], batch_size=16)
develop_dataloader = DataLoader(encoded_dataset["development"], batch_size=16)


main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("main_device: ", main_device)

model = MultipleChoiceModel().to(main_device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch in tqdm(dataloader):
        y = batch["labels"].to(main_device)
        X = []
        for x in ['input_ids', 'attention_mask', 'token_type_ids']:
            src = torch.stack((batch[x][0], batch[x][1], batch[x][2], batch[x][3]), 0).unsqueeze(0)
            X.append(src.permute(1, 0, 2, 3).contiguous())
        X = torch.cat(X, dim=1).to(main_device) # torch.Size([4, 3, batch, MAX_LEN])
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model):
    size = len(dataloader.dataset)
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            y = batch["labels"].to(main_device)
            X = []
            for x in ['input_ids', 'attention_mask', 'token_type_ids']:
                src = torch.stack((batch[x][0], batch[x][1], batch[x][2], batch[x][3]), 0).unsqueeze(0)
                X.append(src.permute(1, 0, 2, 3).contiguous())
            X = torch.cat(X, dim=1).to(main_device) # torch.Size([4, 3, batch, MAX_LEN])
            pred = model(X)
            predictions = torch.argmax(pred, dim=-1)
            correct += (predictions == y).type(torch.float).sum().item()
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}")
    train(train_dataloader, model, loss_fn, optimizer)
    test(develop_dataloader, model)
print("DONE")

torch.save(model.state_dict(), f'{model_name}_trained_model/{model_name}_trained_model.pt')