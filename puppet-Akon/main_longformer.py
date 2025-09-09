import os
import torch.utils.checkpoint as checkpoint
import pandas as pd
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, LongformerConfig
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import  DataLoader
from torch.utils.data import random_split



print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))





# After (explicitly specify use_reentrant)

# Load the dataset
file_path = './scripts_with_labels_preprocessed_final.csv'
dfn = pd.read_csv(file_path)



os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['BUS_ID'] = '00000000:0E:00.0'

# Define a custom dataset class
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

# Initialize tokenizer and model
model_name = 'allenai/longformer-base-4096'
max_len = 4096
new_attention_window =  1024
config = LongformerConfig.from_pretrained(model_name,vocab_size=50265 )
tokenizer = LongformerTokenizerFast.from_pretrained(model_name,attention_window=new_attention_window)
model = LongformerForSequenceClassification.from_pretrained(model_name,config=config)
 


gpu_index = 0  # Change this to the desired GPU index (e.g., 0 for the first GPU)




device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Move your model and data to the device


# Create dataset
dataset = CodeDataset(dfn, tokenizer, max_len)


#print  config 
print(config)

# Move the dataset to the device



# Check if CUDA is available


# Set the device to GPU 0



# Check if CUDA is available


# Create dataset


# Split dataset into train and test
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, tes_dataset,val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size,val_size])


# Create a DataLoader for the train, validation, and test datasets



# Split the dataset

# Create dataloaders

# Check if a GPU is available and set FP16 accordingl
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=40,  # Set a higher number, but use early stopping
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=1000,
    logging_dir='./logs',
    logging_steps=700,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=1e-5,
    weight_decay=0.001,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing_kwargs={'use_reentrant':False}
)

# Define compute_metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Initialize Trainer with Early Stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=7)]  # Stop if no improvement for 3 epochs
)

# Train the model
trainer.train()
# Evaluate the model
results = trainer.evaluate()
print("Test set evaluation results:", results)

# Save the model
trainer.save_model('./best_model')

model.eval()
print("test data") 
trainer.predict(tes_dataset)