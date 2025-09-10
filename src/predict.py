import torch.nn.functional as F
from src.train import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.settings import *
import pickle

"""
This module provides functions to predict the category of a product based on its name and description
using a fine-tuned DistilBERT model for sequence classification.
Functions:
    predict(text):
        Predicts the category index for the given input text using the loaded DistilBERT model.
        Args:
            text (list of str): List containing the input text(s) to classify.
        Returns:
            int: Predicted category index.
    main_predict(name, description):
        Predicts the category label for a product given its name and description.
        Args:
            name (str): The name of the product.
            description (str): The description of the product.
        Returns:
            str: Predicted category label (decoded from label encoder).
"""
# Load model and tokenizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model = model.to(device)
with open(label_encoder_folder+label_encoder_name, 'rb') as pickle_file:
    # Load label encoder
    le = pickle.load(pickle_file)

def predict(text):
    # Tokenize input text
    review_tokenised = tokenizer(text, truncation=True, padding=True)
    # Create a dummy Dataset and DataLoader
    review_dataset = Dataset(review_tokenised, [0])
    # Batch size of 1 for prediction
    review_loader = DataLoader(review_dataset, batch_size=1)
    # Put model in evaluation mode
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
    # Combine name and description into a single text input
    text = str(name) + " " + str(description)
    pred = predict([text])
    category_value = le.inverse_transform([pred])[0]

    return category_value
    
