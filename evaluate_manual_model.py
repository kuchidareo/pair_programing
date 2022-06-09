from transformers import AutoTokenizer, AutoModel
from datasets import Features, load_dataset, Value
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Choose from ["tohoku_bert", "roberta"]
model_mode = "roberta"

BATCH_SIZE = 32
MAX_LENGTH = 80

gpu_device_name = "cuda:0"

data_dir = {"test": "./KUCI/test.jsonl"}
dataset_features=Features({
        "id": Value("int64"),
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
        self.bert = AutoModel.from_pretrained(import_model_name)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(768, 1)
        self.num_choices = num_choices

    def forward(self, x):
        t1 = self.bert(**x)
        t2 = self.dropout(t1.pooler_output)
        logits = self.linear(t2)
        logits = logits.view(-1, self.num_choices)
        return logits
            

def generate_dataloader():
    dataset = load_dataset("json", data_files={"test": data_dir["test"]}, features=dataset_features)
    encoded_dataset = dataset.map(data_to_tensor_features, batched=True)
    # 2 - columns - data_size - 4 - 80

    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
    # 2 - columns - data_size - 4 - 80

    test_dataloader = DataLoader(encoded_dataset["test"], batch_size=BATCH_SIZE)
    # (1) - columns - 4 - data_size - 80

    return {"test": test_dataloader}


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
        add_special_tokens=False
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
print(main_device)


if model_mode == "tohoku_bert":
    import_model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    tokenizer = AutoTokenizer.from_pretrained(import_model_name)
    model = MultipleChoiceModel()
    model.load_state_dict(torch.load(f'{model_mode}_trained_model/{model_mode}_trained_model.pt'))
    model.to(main_device)
elif model_mode == "roberta":
    import_model_name = "rinna/japanese-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(import_model_name)
    model = MultipleChoiceModel()
    model.load_state_dict(torch.load(f'{model_mode}_trained_model/{model_mode}_trained_model.pt'))
    model.to(main_device)


data_loaders = generate_dataloader()
test_dataloader = data_loaders["test"]


def get_predictions(dataloader, model):
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            X = batch_transform(batch, model_mode)
            pred = model(X)
            predictions = torch.argmax(pred, dim=-1).reshape(-1)
            preds.append(predictions)
    preds = torch.cat(preds).cpu().numpy().astype(str)
    return preds


preds = get_predictions(test_dataloader, model)

correspond = {"0": "a", "1": "b", "2": "c", "3": "d"}
for i, label in correspond.items():
    np.place(preds, preds==i, label)

np.savetxt(f"./{model_mode}_trained_model/{model_mode}_prediction.csv", preds, fmt="%s")
