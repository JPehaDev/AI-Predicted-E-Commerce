import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import pickle
from src.settings import *


def read_data():
    products = pd.read_json(products_path, compression="gzip")
    return products

def transform(products):
    products["category_id"] = products["category"].apply(lambda x: x[-1]["id"])
    unique_category = products["category_id"].value_counts(normalize=False)
    valid_category = unique_category[unique_category>=min_value]
    products["label"] = products["category_id"].where(products["category_id"].isin(valid_category.index), other_cat_value)
    features = ["name", "description"]
    for feature, type in products[features].dtypes.to_dict().items():
        products[feature] = products[feature].fillna(fill_nan_values_string)
    products["text"] = products["name"] + " " + products["description"]

    return products

def encoder(products):
    le = LabelEncoder()
    le.fit(products["label"])
    products["label"] = le.transform(products["label"])
    data = products["text"]
    label = products["label"]
    if not os.path.exists(label_encoder_folder):
        os.makedirs(label_encoder_folder)
    output = open(label_encoder_folder+label_encoder_name, 'wb')
    pickle.dump(le, output)
    output.close()

    return data, label

def save_data(data, label):

    assert data.shape[0] == label.shape[0]
    assert data.isna().sum().sum() == 0
    assert label.isna().sum().sum() == 0
    if not os.path.exists(data_pre_processing_folder):
        os.makedirs(data_pre_processing_folder)
    data.to_csv(data_pre_processing_folder+data_pre_processing_name, index=None)
    label.to_csv(data_pre_processing_folder+label_pre_processing_name, index=None)

def main_preprocessdata():
    products = read_data()
    products = transform(products)
    data, label = encoder(products)
    save_data(data, label)
    print(f"Data save in {data_pre_processing_folder}")
