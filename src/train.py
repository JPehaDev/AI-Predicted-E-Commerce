import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from tqdm.notebook import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from src.preprocessdata import data_pre_processing_folder, data_pre_processing_name, label_pre_processing_name
import numpy as np
from src.settings import *

def load_data():
    data = pd.read_csv(data_pre_processing_folder + data_pre_processing_name)["text"]
    label = pd.read_csv(data_pre_processing_folder + label_pre_processing_name)["label"]

    return data, label

def split_data(data, label):
    X_train, X_test, y_train, y_test = train_test_split(data, label, stratify=label, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25, random_state=1)

    return X_train, y_train, X_test, y_test, X_val, y_val

def get_tokens(X_train, X_val, X_test):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_base_name)

    X_train = tokenizer(X_train.to_list(), truncation=True, padding=True, max_length=max_length)
    X_val = tokenizer(X_val.to_list(), truncation=True, padding=True, max_length=max_length)
    X_test = tokenizer(X_test.to_list(), truncation=True, padding=True, max_length=max_length) 

    return X_train, X_val, X_test, tokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def train(num_labels, train_dataset):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DistilBertForSequenceClassification.from_pretrained(model_base_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    model.to(device)
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optim = AdamW(model.parameters(), lr=lr)

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
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation(model, validation_dataloader):
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
    
    val_loader = DataLoader(val_dataset, batch_size=16)
    mean_accuracy, preds_list = validation(model, val_loader)
    y_pred = np.concatenate(list(i.cpu().numpy() for i in preds_list), axis=0)
    weighted, micro, macro = f1_score(y_val, y_pred, average='weighted'), f1_score(y_val, y_pred, average='micro'), f1_score(y_val, y_pred, average='macro')
    print(f"Validation metrics mean_accuracy: {mean_accuracy}, f1 weighted: {weighted}, f1 micro: {micro}, f1 macro: {macro}")

    test_loader = DataLoader(test_dataset, batch_size=16)
    mean_accuracy, preds_list = validation(model, test_loader)
    y_pred = np.concatenate(list(i.cpu().numpy() for i in preds_list), axis=0)
    weighted, micro, macro = f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='macro')
    print(f"Test metrics mean_accuracy: {mean_accuracy}, f1 weighted: {weighted}, f1 micro: {micro}, f1 macro: {macro}")


def main_train():

    data, label = load_data()
    num_labels = label.nunique()
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(data, label)
    X_train, X_val, X_test, tokenizer = get_tokens(X_train, X_val, X_test)

    tokenizer.save_pretrained(tokenizer_path)

    train_dataset = Dataset(X_train, y_train.to_list())
    val_dataset = Dataset(X_val, y_val.to_list())
    test_dataset = Dataset(X_test, y_test.to_list())

    model = train(num_labels, train_dataset)
    model.save_pretrained(model_path)

    evaluate(model, val_dataset, y_val, test_dataset, y_test)
    
    print(f"Save tokenize in {tokenizer_path}")
    print(f"Save model in {model_path}")