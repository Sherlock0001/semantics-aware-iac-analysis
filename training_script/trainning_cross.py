import pandas as pd
from transformers import LongformerForSequenceClassification, LongformerTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
from transformers import EarlyStoppingCallback
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the dataset
dfn = pd.read_csv('./scripts_with_labels_preprocessed_final.csv')

# Initial stratified split to separate test set
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(strat_split.split(dfn, dfn["Defect.Label"]))
train_val_dfn = dfn.iloc[train_val_idx]
test_dfn = dfn.iloc[test_idx]

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
        text = str(self.text[index])
        labels = self.labels[index]
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
max_len = 4096
new_attention_window =  1024

config = LongformerConfig.from_pretrained(model_name,vocab_size=50265 )
config.max_position_embeddings = max_len
config.attention_window = [new_attention_window] * config.num_hidden_layers  # Set attention window for each layer

model = LongformerForSequenceClassification.from_pretrained(model_name,config=config)
 


# Initialize tokenizer with custom attention window
tokenizer = LongformerTokenizerFast.from_pretrained(model_name, attention_window=new_attention_window)


# Initialize the stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=30,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

# Function to compute metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Perform stratified k-fold cross-validation
for fold, (train_index, val_index) in enumerate(skf.split(train_val_dfn, train_val_dfn["Defect.Label"])):
    print(f"Training fold {fold+1}...")

    # Split train_val_dfn into training and validation sets for this fold
    train_fold_df = train_val_dfn.iloc[train_index]
    val_fold_df = train_val_dfn.iloc[val_index]

    # Further split into datasets
    train_dataset = CodeDataset(train_fold_df, tokenizer, max_len)
    val_dataset = CodeDataset(val_fold_df, tokenizer, max_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=7)]  # Stop if no improvement for 3 epochs
    )
    trainer.train()
    results = trainer.evaluate()
    print(f"Fold {fold+1} results: {results}")

    # Initialize the test dataset for evaluation after each fold
 

    # Perform multiple predictions on the test set

num_predictions = 3
for i in range(num_predictions):
    # Final evaluation on the test set after all folds are completed
    print("Final evaluation on the test set...")
    test_dataset = CodeDataset(test_dfn, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    test_results = trainer.predict(test_loader)
    print("Test set final predictions: ", test_results.metrics)
