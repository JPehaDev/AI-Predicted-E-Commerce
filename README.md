# E-Commerce-AI
Repository for AnyoneAI final project 

```bash
git clone https://github.com/SantiCinotti/AnyoneAI-ECommerce

git lfs install

git lfs track "*.safetensors"

git add .gitattributes
```


## App

Run:

```bash
sudo docker-compose up
```


API:
[http://0.0.0.0:8080/](http://0.0.0.0:8080/)


Labs:
[http://localhost:8888/](http://localhost:8888/)


**Model metrics**:

- Validation metrics 
    -   mean_accuracy: 0.910893976688385
    -   f1 weighted: 0.9112792950366866
    -   f1 micro: 0.9108335753703167
    -   f1 macro: 0.8696858562343931

- Test metrics 
    -   mean_accuracy: 0.9111841917037964
    -   f1 weighted: 0.9105223059330424
    -   f1 micro: 0.9111326234269119
    -   f1 macro: 0.8669798601511931


---

##  Challenge

Overview
In a nutshell, this project will result in an API service that is backed by an NLP model that will classify the textual elements of a product record (description, summary etc). The individual components for this project are very similar to what you have implemented in the last three Projects. You can reuse as much code as you want to leave yourself more time for working on the core dataset and model training.


Deliverables 
Goal: The main objective of this project is to build a system/service that will accept a typical "new product"  to classify the product into a set of predefined categories.

In order to graduate from the ML Developer Career, you have to approve the Main Deliverables. You are also welcome to complete the Optional Deliverables if you want to continue to add experience and build your portfolio, although those are not mandatory. 


Main Deliverables:

Exploratory Dataset Analysis (EDA) Jupyter notebooks and dataset
Scripts used for data pre-processing and data preparation
Training scripts and trained models. Description of how to reproduce results
Implementation and training of the NLP model for product classification
API with a basic UI interface for demo (upload of textual description and return of predictions)
Everything must be containerized using Docker
Optional Objectives:

Retraining of model with new data added by users.


Approach and Milestones
There are many ways to approach this project and at first sight, this might seem very overwhelming. A good rule of thumb is this:

Get a good overview and idea of what you need to build.
Identify the unknowns and, most importantly, the major risks for the project and allocate time appropriately. For instance: setting up a dockerized API is simple and low risk but dealing with a dirty dataset is a high-risk task. 
Some tasks will take a long time and might be a blocker for other tasks. Try anticipating the unknowns and allocate time as necessary.



A possible approach is this milestone/project plan:

1.Setup repository and project structure
Create a Github repo. Organize the project, create sub-folders, and prep/mock as much project structure necessary for the final project deliverables. Prepare the AWS server for dataset evaluation.

2.Download and evaluate the dataset
Start with EDA over our dataset. Get metrics about the number of products, unique categories, histogram of categories, etc. Consider that one product can belong to more than one category because there is a taxonomic structure. 
Find the most common categories, using a histogram and removing all categories with a value less than 100. Use the categories above the threshold as the final categories and label every other below as a new “other” category.

3.Create a training dataset
Implement the cleaning step to remove non-alphabetical characters, punctuation, stop words, etc. and encapsulate all of them into a preprocessing function. Use this function to clean the name and description of products, because you will use them as input to our system.
Create your dataset in S3 and save cleaned text with category information.

4.State-of-the-art review
Investigate tokenizers, word embeddings, and TF-IDF to convert text into numbers. 
Investigate different classifiers such as LightGBM, Xgboost, Catboot, RandomForest, Ensemble, and stacked variants.

5.Classifier and accuracy research
Train classifiers that you feel best fit your needs, such as LightGBM, Xgboost, Catboot, RandomForest, Ensemble, and stacked variants. Additionally, train an MLP classifier.

6.Evaluate/test initial classifier
Compare performance (accuracy, AUC, training time, inference time, etc) of previous classifiers vs MLP classifiers using cleaned data.
Choose the best model according to experiments prioritizing AUC and inference time.

7.Setup an API for product classification
It’s time to put the model into production. Our goal is to build an API service as we did for Sprint 4 so we make this model accessible to other components. Although not mandatory, instead of Flask, this time you can use FastAPI (one of the most used frameworks to deploy ML models these days). The API should be containerized using Docker.

8.Integrate a basic UI and secure the API
Create a basic UI that can drive the API service and allow the user to submit the product name and description data and get a category prediction as a result. It should have a web UI for making demos but also an endpoint so others can use our API to interact with our model and integrate it with third-party services. 

9.Fine-tune model / Train additional models
After the evaluation of the initial classifier, accuracy can be improved. Keep training more models and see if you can improve the baseline accuracy. 

9,5.Add API tests (Optional)
You should add some tests to the API main components.

10.Preview service to other teams
Demo the project to other teams and gather their feedback for final adjustments.

11.Build final presentation and prep for demo
Build your presentation and prepare your project for the Demo Day
Keep in mind that the suggested plan and milestones are high-level and you will most likely need to break it down further into smaller tasks and extended it as needed based on your discovery and/or issues that your run into.




Dataset
The dataset for this project is based on the open e-commerce dataset from BestBuy.com (largest online and brick-and-mortar retailer in the US). The dataset can be downloaded here.

The dataset contains two source files: categories.json and products.json which will be relevant to this project. Both files are small enough to be accessed directly from Github using the following raw URL schema:

https://raw.githubusercontent.com/anyoneai/e-commerce-open-data-set/master/products.json

Here's an example of an entry for a single product:

{
        "sku": 1004695,
        "name": "GoPro - Camera Mount Accessory Kit - Black",
        "type": "HardGood",
        "price": 19.99,
        "upc": "185323000309",
        "category":
        [
            {
                "id": "abcat0400000",
                "name": "Cameras & Camcorders"
            },
            {
                "id": "abcat0410022",
                "name": "Camcorder Accessories"
            },
            {
                "id": "pcmcat329700050009",
                "name": "Action Camcorder Accessories"
            },
            {
                "id": "pcmcat240500050057",
                "name": "Action Camcorder Mounts"
            },
            {
                "id": "pcmcat329700050020",
                "name": "Handlebar/Seatpost Mounts"
            }
        ],
        "shipping": 5.49,
        "description": "Compatible with most GoPro cameras; includes a variety of camera mounting accessories",
        "manufacturer": "GoPro",
        "model": "AGBAG-001",
        "url": "http://www.bestbuy.com/site/gopro-camera-mount-accessory-kit-black/1004695.p?id=1218249514954&skuId=1004695&cmp=RMXCC",
        "image": "http://img.bbystatic.com/BestBuy_US/images/products/1004/1004695_rc.jpg"
  }
		
 
However, you might also realize that "simple" is a two-edged sword. In the above example, the description is very short, not much to work with. The is, unfortunately, the current state of the industry and not uncommon (here is an example of a really bad description from Amazon) and hopefully, your model will be able to deal with that.

References
There are quite a few references and existing models on the internet. Feel free to research as much as you need on the subject. Below you will find a couple of good starting points:

Large Scale Product Categorization using Structured and Unstructured Attributes - Abhinandan Krishnan, Abilash Amarthaluri
Multi-Label Product Categorization Using Multi-Modal Fusion Models - Pasawee Wirojwatanakul, Artit Wangperawong
