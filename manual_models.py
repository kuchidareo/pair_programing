from transformers import AutoTokenizer, AutoModel, BertConfig, BertTokenizer, BertModel
from datasets import Features, load_dataset, ClassLabel, Value
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Choose from ["tohoku_bert", "kyodai_bert", "roberta"]
model_mode = "tohoku_bert"

MAX_LENGTH = 80
BATCH_SIZE = 16

gpu_device_name = "cuda:0"

small_data_mode = False
if small_data_mode:
    data_dir = {
        "train": "./KUCI/small_train.jsonl",
        "development": "./KUCI/small_development.jsonl"
    }
else:
    data_dir = {
        "train": "./KUCI/train.jsonl",
        "development": "./KUCI/development.jsonl"
    }


dataset_features = Features({
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
    def __init__(self, num_choices=4):
        super(MultipleChoiceModel, self).__init__()
        self.bert = AutoModel.from_pretrained(import_model_name, config=config)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(768, 1)
        self.num_choices = num_choices

    def forward(self, x):
        t1 = self.bert(**x)
        t2 = self.dropout(t1.pooler_output)
        logits = self.linear(t2)
        logits = logits.view(-1, self.num_choices)
        return logits


def generate_dataloader(model_mode):
    dataset = load_dataset("json", data_files={"train": data_dir["train"], "development": data_dir["development"]}, features=dataset_features)
    encoded_dataset = dataset.map(data_to_tensor_features, batched=True)
    # 2 - colums - data_size - 4 - 80

    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    if model_mode == "roberta":
        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    else:
        encoded_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    # 2 - colums - data_size - 4 - 80

    train_dataloader = DataLoader(encoded_dataset["train"], batch_size=BATCH_SIZE)
    develop_dataloader = DataLoader(encoded_dataset["development"], batch_size=BATCH_SIZE)
    # (1) - colums - 4 - data_size - 80

    return {"train": train_dataloader, "develop": develop_dataloader}
            

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
        padding="max_length", 
        max_length=MAX_LENGTH, 
        truncation=True,
        add_special_tokens=False,
    )

    # k = <str> e.g, "input_ids"
    # v = <Tensor> len(v) = 4 * len(context)
    features = {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_connected_sentences.items()}
    return features


def batch_transform(batch, model_mode):
    if model_mode == "roberta":
        input_list = ['input_ids', 'attention_mask']
    else:
        input_list = ['input_ids', 'attention_mask', 'token_type_ids']

    y = batch["labels"].to(main_device)

    # ori: columns(dict) - 4(list) - tensor(batch - 80) を
    # X  : columns(dict) - tensor(batch*4 - 80) にしたい。
    X = {}
    for x in input_list:
        src = torch.stack((batch[x][0], batch[x][1], batch[x][2], batch[x][3]), 0)
        src = src.permute(1, 0, 2).contiguous()
        X[x] = src.view(-1, src.size(-1)).to(main_device)

    return X, y

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


main_device = torch.device(gpu_device_name if torch.cuda.is_available() else "cpu")
print("main_device: ", main_device)


if model_mode == "tohoku_bert":
    import_model_name = "cl-tohoku/bert-base-japanese"
    config = None
    tokenizer = AutoTokenizer.from_pretrained(import_model_name)
elif model_mode == "kyodai_bert":
    import_model_name = "../preTrainedModels/bert/Japanese_L-12_H-768_A-12_E-30_BPE_transformers"
    config = BertConfig.from_json_file(f'{import_model_name}/config.json')
    tokenizer = BertTokenizer(f'{import_model_name}/vocab.txt', do_lower_case=False, do_basic_tokenize=False)
elif model_mode == "roberta":
    import_model_name = "rinna/japanese-roberta-base"
    config = None
    tokenizer = AutoTokenizer.from_pretrained(import_model_name)

model = MultipleChoiceModel().to(main_device)
data_loaders = generate_dataloader(model_mode)
train_dataloader = data_loaders["train"]
develop_dataloader = data_loaders["develop"]

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch in tqdm(dataloader):
        X, y = batch_transform(batch, model_mode)
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
            X, y = batch_transform(batch, model_mode)
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
    if not small_data_mode:
        torch.save(model.state_dict(), f'{model_mode}_trained_model/{model_mode}_trained_model_epoch{t}.pt')
        print(f"model saved at: {model_mode}_trained_model/{model_mode}_trained_model_epoch{t}.pt")
print("DONE")

