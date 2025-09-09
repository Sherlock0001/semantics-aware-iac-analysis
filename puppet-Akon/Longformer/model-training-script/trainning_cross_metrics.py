from sklearn.model_selection import train_test_split,KFold, StratifiedShuffleSplit
import time
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

# Load the dataset
dfn = pd.read_csv('./scripts_with_labels_preprocessed_final.csv')
print("Length of dfn:", len(dfn))

# Initial stratified split to separate test set
random_state = int(time.time())
train_val_dfn, test_dfn = train_test_split(dfn, test_size=0.2, random_state=random_state, stratify=dfn["Defect.Label"])

# Define the dataset class
class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe["Script.Content"]
        self.labels = dataframe["Defect.Label"]
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

# Initialize the model and tokenizer
model_name = 'allenai/longformer-base-4096'
max_len = 1500
model = LongformerForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = LongformerTokenizerFast.from_pretrained(model_name, max_length=max_len)

# Initialize the stratified k-fold cross-validation
kf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
another_number = torch.randint(0, 10000, (1,)).item()  # Generate a random integer between 0 and 100
combined_seed = int(time.time()) + another_number

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_using_8-folds_16_batch_1500token',
    num_train_epochs=40,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    fp16=True,
    dataloader_num_workers=accelerator.num_processes
)

# Initialize optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=1000, num_training_steps=10000
)

# Function to compute metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    auc = roc_auc_score(p.label_ids, p.predictions[:, 1])
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': acc
    }

# Initialize lists to store the metrics
auc_list = []
precision_list = []
recall_list = []
f1_list = []
accuracy_list = []

# Perform stratified k-fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(train_val_dfn, train_val_dfn["Defect.Label"])):
    print(f"Training fold {fold+1}...")
    
    # Split train_val_dfn into training and validation sets for this fold
    train_dfn, val_dfn = train_val_dfn.iloc[train_index], train_val_dfn.iloc[val_index]
    
    # Initialize datasets for training and validation
    train_dataset = CodeDataset(train_dfn, tokenizer, max_len)
    val_dataset = CodeDataset(val_dfn, tokenizer, max_len)
    
    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
    )
    trainer.train()
    results = trainer.evaluate()
    print(f"Fold {fold+1} results: {results}")
    trainer.save_model(f'./best_model_fold_1500token_16batch')
    
    # Evaluate the model on the test set
    test_dataset = CodeDataset(test_dfn, tokenizer, max_len)
    test_results = trainer.predict(test_dataset)
    print(f"Fold {fold+1} Test set final predictions: {test_results.metrics}")
    
    # Store the metrics
    auc_list.append(test_results.metrics['test_auc'])
    precision_list.append(test_results.metrics['test_precision'])
    recall_list.append(test_results.metrics['test_recall'])
    f1_list.append(test_results.metrics['test_f1'])
    accuracy_list.append(test_results.metrics['test_accuracy'])

# Print summary statistics for each metric
print("*************************")
print(f"Summary: AUC, mean: {np.mean(auc_list):.4f}, median: {np.median(auc_list):.4f}, max: {np.max(auc_list):.4f}, min: {np.min(auc_list):.4f}")
print("*************************")
print(f"Summary: Precision, mean: {np.mean(precision_list):.4f}, median: {np.median(precision_list):.4f}, max: {np.max(precision_list):.4f}, min: {np.min(precision_list):.4f}")
print("*************************")
print(f"Summary: Recall, mean: {np.mean(recall_list):.4f}, median: {np.median(recall_list):.4f}, max: {np.max(recall_list):.4f}, min: {np.min(recall_list):.4f}")
print("*************************")
print(f"Summary: F1 Score, mean: {np.mean(f1_list):.4f}, median: {np.median(f1_list):.4f}, max: {np.max(f1_list):.4f}, min: {np.min(f1_list):.4f}")
print("*************************")
print(f"Summary: Accuracy, mean: {np.mean(accuracy_list):.4f}, median: {np.median(accuracy_list):.4f}, max: {np.max(accuracy_list):.4f}, min: {np.min(accuracy_list):.4f}")
print("*************************")
