import torch.nn.functional as F
from src.train import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.settings import *
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model = model.to(device)
with open(label_encoder_folder+label_encoder_name, 'rb') as pickle_file:
    le = pickle.load(pickle_file)

def predict(text):
    
    review_tokenised = tokenizer(text, truncation=True, padding=True)
    review_dataset = Dataset(review_tokenised, [0])
    review_loader = DataLoader(review_dataset, batch_size=1)
    with torch.no_grad():
        for batch in review_loader : 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            prediction = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = F.softmax(prediction[1], dim=1)

    _,preds = torch.max(logits, dim=1) 

    return preds.item()

def main_predict(name, description):

    text = str(name) + " " + str(description)
    pred = predict([text])
    category_value = le.inverse_transform([pred])[0]

    return category_value
    
