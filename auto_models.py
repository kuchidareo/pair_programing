from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from datasets import Features, load_dataset, ClassLabel, Value
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import numpy as np
import torch

# Choose from ["auto_tohoku_bert", "auto_roberta"]
model_mode = "auto_roberta"
BATCH_SIZE = 8

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

# adding [PAD] to each batch
@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


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
        truncation=True,
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


device = torch.device(gpu_device_name if torch.cuda.is_available() else "cpu")
print(device)

if model_mode == "auto_tohoku_bert":
    import_model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    tokenizer = AutoTokenizer.from_pretrained(import_model_name)
    model = AutoModelForMultipleChoice.from_pretrained(import_model_name).to(device)
elif model_mode == "auto_roberta":
    import_model_name = "rinna/japanese-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(import_model_name)
    model = AutoModelForMultipleChoice.from_pretrained(import_model_name).to(device)


dataset = load_dataset("json", data_files={"train": data_dir["train"], "development": data_dir["development"]}, features=dataset_features)
encoded_dataset = dataset.map(data_to_tensor_features, batched=True)


args = TrainingArguments(
    f"{model_mode}",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["development"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
if not small_data_mode:
    model.save_pretrained(f'{model_mode}_trained_model')
