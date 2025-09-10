import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import pickle
from src.settings import *

"""
This module provides functions to preprocess product data for an e-commerce application.
It includes reading data, transforming features, encoding labels, and saving the processed data.
Functions:
    read_data():
        Reads the products data from a JSON file specified by `products_path` with gzip compression.
        Returns:
            pd.DataFrame: The loaded products data.
    transform(products):
        Transforms the products DataFrame by extracting the last category ID, filtering categories
        with insufficient samples, assigning a default value to rare categories, filling NaN values,
        and creating a combined text feature from 'name' and 'description'.
        Args:
            products (pd.DataFrame): The products data.
        Returns:
            pd.DataFrame: The transformed products data.
    encoder(products):
        Encodes the 'label' column using sklearn's LabelEncoder, saves the encoder as a pickle file,
        and returns the text and encoded label columns.
        Args:
            products (pd.DataFrame): The transformed products data.
        Returns:
            Tuple[pd.Series, pd.Series]: The text data and encoded labels.
    save_data(data, label):
        Saves the processed text and label data as CSV files in the specified folder.
        Args:
            data (pd.Series): The text data.
            label (pd.Series): The encoded labels.
    main_preprocessdata():
        Orchestrates the preprocessing pipeline: reads data, transforms it, encodes labels, saves the results,
        and prints the save location.
"""

def read_data():
    products = pd.read_json(products_path, compression="gzip")
    return products

def transform(products):
    # Get the last category in the list
    products["category_id"] = products["category"].apply(lambda x: x[-1]["id"])
    # Account for categories with few samples
    unique_category = products["category_id"].value_counts(normalize=False)
    # Filter categories with less than min_value samples
    valid_category = unique_category[unique_category>=min_value]
    # If category_id is not in valid_category, set it to other_cat_value
    products["label"] = products["category_id"].where(products["category_id"].isin(valid_category.index), other_cat_value)
    # Create text feature
    features = ["name", "description"]
    # Fill NaN values with a string
    for feature, type in products[features].dtypes.to_dict().items():
        # Concatenate feature name with type
        products[feature] = products[feature].fillna(fill_nan_values_string)
    products["text"] = products["name"] + " " + products["description"]
    #Return only text and label columns
    return products

def encoder(products):
    # Encode labels
    le = LabelEncoder()
    # Fit and transform the label column
    le.fit(products["label"])
    #Replace label column with encoded values
    products["label"] = le.transform(products["label"])
    #Separate data and label
    data = products["text"]
    label = products["label"]
    # Save the label encoder
    if not os.path.exists(label_encoder_folder):
        os.makedirs(label_encoder_folder)
    output = open(label_encoder_folder+label_encoder_name, 'wb')
    # Save the label encoder
    pickle.dump(le, output)
    output.close()
    # Return data and label
    return data, label

def save_data(data, label):
    # Check that data and label have the same length and no NaN values
    assert data.shape[0] == label.shape[0]
    # Check that data and label have the same length and no NaN values
    assert data.isna().sum().sum() == 0
    assert label.isna().sum().sum() == 0
    # Save data and label as csv files
    if not os.path.exists(data_pre_processing_folder):
        os.makedirs(data_pre_processing_folder)
    data.to_csv(data_pre_processing_folder+data_pre_processing_name, index=None)
    label.to_csv(data_pre_processing_folder+label_pre_processing_name, index=None)

def main_preprocessdata():
    # Run the preprocessing pipeline
    products = read_data()
    products = transform(products)
    data, label = encoder(products)
    save_data(data, label)
    print(f"Data save in {data_pre_processing_folder}")
