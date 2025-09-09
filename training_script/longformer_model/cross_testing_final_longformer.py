from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from accelerate import Accelerator
from sklearn.model_selection import train_test_split



# Initialize accelerator
accelerator = Accelerator()

# Load the dataset
dfn = pd.read_csv('./scripts_with_labels_preprocessed_final.csv')
print("Length of dfn:", len(dfn))
train_val_dfn_main, test_dfn = train_test_split(dfn, test_size=0.2, random_state=42, stratify=dfn["Defect.Label"])
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
tokenizer = LongformerTokenizerFast.from_pretrained(model_name, max_length=max_len)
model = LongformerForSequenceClassification.from_pretrained(model_name)
# Initialize the stratified k-fold cross-validation
kf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_using_8-folds_8_batch_1500token',
    num_train_epochs=40,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    gradient_accumulation_steps=4,
    weight_decay=0.001,
    fp16=True,
    dataloader_num_workers=accelerator.num_processes
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
val_auc_list = []
val_precision_list = []
val_recall_list = []
val_f1_list = []
val_accuracy_list = []

test_auc_list = []
test_precision_list = []
test_recall_list = []
test_f1_list = []
test_accuracy_list = []

# Initialize optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=1000, num_training_steps=10000
)


# Perform stratified k-fold cross-validation
for fold, (train_val_index, test_index) in enumerate(kf.split(train_val_dfn_main, train_val_dfn_main["Defect.Label"])):
    random_seed = 42 + fold 
    print(f"Training fold {fold+1}...")
    # Split into training+validation and test sets
    train_val_dfn, test_dfn = dfn.iloc[train_val_index], dfn.iloc[test_index]
    train_dfn, val_dfn = train_test_split(train_val_dfn, test_size=0.2, random_state= random_seed )

    # Initialize datasets for training, validation, and test
    train_dataset = CodeDataset(train_dfn, tokenizer, max_len)
    val_dataset = CodeDataset(val_dfn, tokenizer, max_len)
    test_dataset = CodeDataset(test_dfn, tokenizer, max_len)
    
    # Initialize DataLoader for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_args.per_device_eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size)
    
    
    
    # Initialize optimizer and scheduler
    

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )
    trainer.train()
    val_results = trainer.evaluate()
    trainer.save_model(f'./best_model_fold_1500token_8batch')
    print(f"Fold {fold+1} validation results: {val_results}")
    
    # Store the validation metrics
    val_auc_list.append(val_results['eval_auc'])
    val_precision_list.append(val_results['eval_precision'])
    val_recall_list.append(val_results['eval_recall'])
    val_f1_list.append(val_results['eval_f1'])
    val_accuracy_list.append(val_results['eval_accuracy'])
    
    # Evaluate the model on the test set
    test_results = trainer.predict(test_dataset)
    print(f"Fold {fold+1} Test set final predictions: {test_results.metrics}")
    
    # Store the test metrics
    test_auc_list.append(test_results.metrics['test_auc'])
    test_precision_list.append(test_results.metrics['test_precision'])
    test_recall_list.append(test_results.metrics['test_recall'])
    test_f1_list.append(test_results.metrics['test_f1'])
    test_accuracy_list.append(test_results.metrics['test_accuracy'])

# Print summary statistics for each metric from the validation set
print("*************************")
print(f"Validation Summary: AUC, mean: {np.mean(val_auc_list):.4f}, median: {np.median(val_auc_list):.4f}, max: {np.max(val_auc_list):.4f}, min: {np.min(val_auc_list):.4f}")
print("*************************")
print(f"Validation Summary: Precision, mean: {np.mean(val_precision_list):.4f}, median: {np.median(val_precision_list):.4f}, max: {np.max(val_precision_list):.4f}, min: {np.min(val_precision_list):.4f}")
print("*************************")
print(f"Validation Summary: Recall, mean: {np.mean(val_recall_list):.4f}, median: {np.median(val_recall_list):.4f}, max: {np.max(val_recall_list):.4f}, min: {np.min(val_recall_list):.4f}")
print("*************************")
print(f"Validation Summary: F1 Score, mean: {np.mean(val_f1_list):.4f}, median: {np.median(val_f1_list):.4f}, max: {np.max(val_f1_list):.4f}, min: {np.min(val_f1_list):.4f}")
print("*************************")
print(f"Validation Summary: Accuracy, mean: {np.mean(val_accuracy_list):.4f}, median: {np.median(val_accuracy_list):.4f}, max: {np.max(val_accuracy_list):.4f}, min: {np.min(val_accuracy_list):.4f}")


# Print summary statistics for each metric from the test set
print("*************************")
print(f"Test Summary: AUC, mean: {np.mean(test_auc_list):.4f}, median: {np.median(test_auc_list):.4f}, max: {np.max(test_auc_list):.4f}, min: {np.min(test_auc_list):.4f}")
print("*************************")
print(f"Test Summary: Precision, mean: {np.mean(test_precision_list):.4f}, median: {np.median(test_precision_list):.4f}, max: {np.max(test_precision_list):.4f}, min: {np.min(test_precision_list):.4f}")
print("*************************")
print(f"Test Summary: Recall, mean: {np.mean(test_recall_list):.4f}, median: {np.median(test_recall_list):.4f}, max: {np.max(test_recall_list):.4f}, min: {np.min(test_recall_list):.4f}")
print("*************************")
print(f"Test Summary: F1 Score, mean: {np.mean(test_f1_list):.4f}, median: {np.median(test_f1_list):.4f}, max: {np.max(test_f1_list):.4f}, min: {np.min(test_f1_list):.4f}")
print("*************************")
print(f"Test Summary: Accuracy, mean: {np.mean(test_accuracy_list):.4f}, median: {np.median(test_accuracy_list):.4f}, max: {np.max(test_accuracy_list):.4f}, min: {np.min(test_accuracy_list):.4f}")



best_model_path = "./best_model_fold_1500token_8batch"  
final_model = LongformerForSequenceClassification.from_pretrained(best_model_path)

# Define the datasets
final_train_dataset = CodeDataset(train_val_dfn_main, tokenizer, max_len)
final_test_dataset = CodeDataset(test_dfn, tokenizer, max_len)

# Reinitialize the Trainer with the loaded model
trainer = Trainer(
    model=final_model,
    args=training_args,
    train_dataset=final_train_dataset,
    eval_dataset=final_test_dataset,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics
)

# Evaluating the final model on the test dataset
test_res = trainer.evaluate(final_test_dataset)
print(f"Final best model test results: {test_res}")
