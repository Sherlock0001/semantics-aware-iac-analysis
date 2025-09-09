from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from accelerate import Accelerator
from sklearn.model_selection import train_test_split



# Initialization of the accelerator
AcceleratorInit = Accelerator()

# Loading the dataset
dfn = pd.read_csv('./scripts_with_labels_preprocessed_final.csv')
print("Length of dfn:", len(dfn))

# Defining the dataset class 
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

#Defining the model name 
model_name = 'allenai/longformer-base-4096'
max_len = 1800
tokenizer = LongformerTokenizerFast.from_pretrained(model_name, max_length=max_len)
model = LongformerForSequenceClassification.from_pretrained(model_name)
# Initialize the stratified k-fold  cross_validation with 8 folds 
kf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_using_8-folds_8_batch_1800token',
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
    dataloader_num_workers=AcceleratorInit.num_processes
)

# Computing  metrics in this function we are computing  auc , recall ,f1 accuracy score 
        
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

# Initializing evalution lists to be able to store it every epoch  
validation_auc_lst = []
validation_prec_lst = []
validation_rec_lst = []
validation_f1_lst = []
validation_acc_lst = []

# Initializing test lists to be able to store it every epoch  
test_auc_lst = []
test_prec_lst= []
test_rec_lst = []
test_f1_lst = []
test_acc_lst= []

# Initialize optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=1000, num_training_steps=10000
)


# Perform stratified k-fold cross-validation
for fold, (train_val_index, test_index) in enumerate(kf.split(dfn, dfn["Defect.Label"])):
    
    print(f"Training fold {fold+1}...")
    # Split into training+validation and test sets
    train_val_dfn, test_dfn = dfn.iloc[train_val_index], dfn.iloc[test_index]
    train_dfn, val_dfn = train_test_split(train_val_dfn, test_size=0.2, random_state=42)

    # Initialize datasets for training, validation, and test
    train_dataset = CodeDataset(train_dfn, tokenizer, max_len)
    val_dataset = CodeDataset(val_dfn, tokenizer, max_len)
    test_dataset = CodeDataset(test_dfn, tokenizer, max_len)
    
    # Initialize DataLoader for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_args.per_device_eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size)
    
    
    
    
    

    # Training the model using Transformer kernal 
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
    val_res = trainer.evaluate()
    print(f"Fold {fold+1} validation results: {val_res}")
    

    # Appending the validation metrics 
    validation_auc_lst.append(val_res['eval_auc'])
    validation_prec_lst.append(val_res['eval_precision'])
    validation_rec_lst.append(val_res['eval_recall'])
    validation_f1_lst.append(val_res['eval_f1'])
    validation_acc_lst.append(val_res['eval_accuracy'])
    
    # Evaluating the model with test dataset at the end of each fold 
    test_res = trainer.predict(test_dataset)
    print(f"Fold {fold+1} Test set final predictions: {test_res.metrics}")
    
     # Appending the test metrics 
    test_auc_lst.append(test_res.metrics['test_auc'])
    test_prec_lst.append(test_res.metrics['test_precision'])
    test_rec_lst.append(test_res.metrics['test_recall'])
    test_f1_lst.append(test_res.metrics['test_f1'])
    test_acc_lst.append(test_res.metrics['test_accuracy'])

#Printing summary statistics for all folds for validation metrics 
print("*************************")
print(f"Validation Summary: AUC, mean: {np.mean(validation_auc_lst):.4f}, median: {np.median(validation_auc_lst):.4f}, max: {np.max(validation_auc_lst):.4f}, min: {np.min(validation_auc_lst):.4f}")
print("*************************")
print(f"Validation Summary: Precision, mean: {np.mean(validation_prec_lst):.4f}, median: {np.median(validation_prec_lst):.4f}, max: {np.max(validation_prec_lst):.4f}, min: {np.min(validation_prec_lst):.4f}")
print("*************************")
print(f"Validation Summary: Recall, mean: {np.mean(validation_rec_lst):.4f}, median: {np.median(validation_rec_lst):.4f}, max: {np.max(validation_rec_lst):.4f}, min: {np.min(validation_rec_lst):.4f}")
print("*************************")
print(f"Validation Summary: F1 Score, mean: {np.mean(validation_f1_lst):.4f}, median: {np.median(validation_f1_lst):.4f}, max: {np.max(validation_f1_lst):.4f}, min: {np.min(validation_f1_lst):.4f}")
print("*************************")
print(f"Validation Summary: Accuracy, mean: {np.mean(validation_acc_lst):.4f}, median: {np.median(validation_acc_lst):.4f}, max: {np.max(validation_acc_lst):.4f}, min: {np.min(validation_acc_lst):.4f}")


# Printing summary statistics for all folds for test metrics 
print("*************************")
print(f"Test Summary: AUC, mean: {np.mean(test_auc_lst):.4f}, median: {np.median(test_auc_lst):.4f}, max: {np.max(test_auc_lst):.4f}, min: {np.min(test_auc_lst):.4f}")
print("*************************")
print(f"Test Summary: Precision, mean: {np.mean(test_prec_lst):.4f}, median: {np.median(test_prec_lst):.4f}, max: {np.max(test_prec_lst):.4f}, min: {np.min(test_prec_lst):.4f}")
print("*************************")
print(f"Test Summary: Recall, mean: {np.mean(test_rec_lst):.4f}, median: {np.median(test_rec_lst):.4f}, max: {np.max(test_rec_lst):.4f}, min: {np.min(test_rec_lst):.4f}")
print("*************************")
print(f"Test Summary: F1 Score, mean: {np.mean(test_f1_lst):.4f}, median: {np.median(test_f1_lst):.4f}, max: {np.max(test_f1_lst):.4f}, min: {np.min(test_f1_lst):.4f}")
print("*************************")
print(f"Test Summary: Accuracy, mean: {np.mean(test_acc_lst):.4f}, median: {np.median(test_acc_lst):.4f}, max: {np.max(test_acc_lst):.4f}, min: {np.min(test_acc_lst):.4f}")