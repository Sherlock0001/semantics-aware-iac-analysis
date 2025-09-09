import torch
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Load the saved model and tokenizer
model_path = './results_using_3072_2048_folds/checkpoint-942'
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
model = LongformerForSequenceClassification.from_pretrained(model_path)

# Set the device to GPU
device = torch.device("cuda:0")

# Load the dataset
file_path = './scripts_with_labels_preprocessed_final.csv'
dfn = pd.read_csv(file_path)
dfn = dfn.reset_index(drop=True)
print(dfn.head())

# Define a custom dataset class
class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['Script.Content']
        self.labels = dataframe['Defect.Label']
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        labels = self.labels.iloc[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

max_len = 4096
# Create dataset
dataset = CodeDataset(dfn, tokenizer, max_len)

# Split dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Define a function to compute metrics

def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    f1_score_value = f1_score(labels, preds, average='weighted')
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "f1_score": f1_score_value
    }

# Prepare the test dataset
test_dataset = CodeDataset(dfn.iloc[train_size:], tokenizer, max_len)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

# Evaluate the model on the test dataset
model.to(device)
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['labels'].to(device)
        }
        outputs = model(**inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(inputs['labels'].cpu().numpy())

# Compute metrics
metrics = compute_metrics(all_preds, all_labels)
print("Test set evaluation results:", metrics)
