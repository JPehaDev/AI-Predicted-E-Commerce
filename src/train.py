import pandas as pd
# Split data
from sklearn.model_selection import train_test_split
# PyTorch
import torch
# Progress bar
from tqdm.notebook import tqdm
# Neural network functions
import torch.nn.functional as F
# Metrics
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
# HuggingFace
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
# Csv file operations
from src.preprocessdata import data_pre_processing_folder, data_pre_processing_name, label_pre_processing_name
import numpy as np
# Hyperparameters and paths
from src.settings import *

"""
This module provides functions and classes for training and evaluating a DistilBERT-based sequence classification model using PyTorch and HuggingFace Transformers.
Functions:
    load_data():
        Loads preprocessed text data and labels from CSV files.
    split_data(data, label):
        Splits the data and labels into training, validation, and test sets with stratification.
    get_tokens(X_train, X_val, X_test):
        Tokenizes the input text datasets using DistilBERT tokenizer.
    train(num_labels, train_dataset):
        Trains a DistilBERT sequence classification model on the provided training dataset.
    accuracy(outputs, labels):
        Computes the accuracy of model predictions against true labels.
    validation(model, validation_dataloader):
        Evaluates the model on a validation dataset, returning mean accuracy and predictions.
    evaluate(model, val_dataset, y_val, test_dataset, y_test):
        Evaluates the model on validation and test datasets, printing accuracy and F1 scores.
    main_train():
        Orchestrates the full training and evaluation pipeline, including data loading, tokenization, training, evaluation, and saving the model and tokenizer.
Classes:
    Dataset(torch.utils.data.Dataset):
        Custom dataset class for handling tokenized inputs and labels for PyTorch DataLoader.
"""

def load_data():
    # Load preprocessed data
    data = pd.read_csv(data_pre_processing_folder + data_pre_processing_name)["text"]
    label = pd.read_csv(data_pre_processing_folder + label_pre_processing_name)["label"]

    return data, label

def split_data(data, label):
    # Split data into train 20%, validation and test 25% sets
    X_train, X_test, y_train, y_test = train_test_split(data, label, stratify=label, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25, random_state=1)
    return X_train, y_train, X_test, y_test, X_val, y_val

    # Tokenize data
def get_tokens(X_train, X_val, X_test):
    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_base_name)
    # Convert to list and tokenize
    X_train = tokenizer(X_train.to_list(), truncation=True, padding=True, max_length=max_length)
    X_val = tokenizer(X_val.to_list(), truncation=True, padding=True, max_length=max_length)
    X_test = tokenizer(X_test.to_list(), truncation=True, padding=True, max_length=max_length) 

    return X_train, X_val, X_test, tokenizer

class Dataset(torch.utils.data.Dataset):
    # Container for our data
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Return item at index idx
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    # Length of dataset
    def __len__(self):
        return len(self.labels)
    

def train(num_labels, train_dataset):
    # Cuda if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Model base + classification head
    model = DistilBertForSequenceClassification.from_pretrained(model_base_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    model.to(device)
    # Train the model
    model.train()
    # Batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Optimizer
    optim = AdamW(model.parameters(), lr=lr)
    # Training loop
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            optim.zero_grad()
    model.eval()

    return model

def accuracy(outputs, labels):
    # Get the predictions
        _, preds = torch.max(outputs, dim=1)
        # Calculate the accuracy
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation(model, validation_dataloader):
    # Validation loop
    with torch.no_grad():
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        loss_val_list = []
        preds_list = []
        accuracy_list = []
        accuracy_sum = 0
        for batch in tqdm(validation_dataloader):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            logits = F.softmax(outputs[1], dim=1)   # Taking the softmax of output
            _,preds = torch.max(logits, dim=1)      # Taking the predictions of our batch
            acc = accuracy(logits,labels)           # Calculating the accuracy of current batch
            accuracy_sum += acc                     # Taking sum of all the accuracies of all the batches. This sum will be divided by batch length to get mean accuracy for validation dataset

            loss_val_list.append(loss)
            preds_list.append(preds)
            accuracy_list.append(acc)

        mean_accuracy = accuracy_sum / len(validation_dataloader)

    return mean_accuracy, preds_list

def evaluate(model, val_dataset, y_val, test_dataset, y_test):
    # Create dataloaders
    val_loader = DataLoader(val_dataset, batch_size=16)
    mean_accuracy, preds_list = validation(model, val_loader)
    y_pred = np.concatenate(list(i.cpu().numpy() for i in preds_list), axis=0)
    # Calculate f1 scores
    weighted, micro, macro = f1_score(y_val, y_pred, average='weighted'), f1_score(y_val, y_pred, average='micro'), f1_score(y_val, y_pred, average='macro')
    print(f"Validation metrics mean_accuracy: {mean_accuracy}, f1 weighted: {weighted}, f1 micro: {micro}, f1 macro: {macro}")
    # Evaluate on test set
    test_loader = DataLoader(test_dataset, batch_size=16)
    mean_accuracy, preds_list = validation(model, test_loader)
    y_pred = np.concatenate(list(i.cpu().numpy() for i in preds_list), axis=0)
    # Calculate f1 scores
    weighted, micro, macro = f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro')
    # Print test metrics
    print(f"Test metrics mean_accuracy: {mean_accuracy}, f1 weighted: {weighted}, f1 micro: {micro}, f1 macro: {macro}")


def main_train():
    # Run the training and evaluation pipeline
    data, label = load_data()
    num_labels = label.nunique()
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(data, label)
    X_train, X_val, X_test, tokenizer = get_tokens(X_train, X_val, X_test)
    # Save the tokenizer
    tokenizer.save_pretrained(tokenizer_path)
    # Create datasets
    train_dataset = Dataset(X_train, y_train.to_list())
    val_dataset = Dataset(X_val, y_val.to_list())
    test_dataset = Dataset(X_test, y_test.to_list())
    # Train the model
    model = train(num_labels, train_dataset)
    model.save_pretrained(model_path)
    # Evaluate the model
    evaluate(model, val_dataset, y_val, test_dataset, y_test)
    # Save the model and tokenizer
    print(f"Save tokenizer in {tokenizer_path}")
    print(f"Save model in {model_path}")