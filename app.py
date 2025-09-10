from fastapi import FastAPI, Form
from pydantic import BaseModel
from src.preprocessdata import read_data
from src.settings import redis_port
from src.predict import main_predict
import redis
from typing import Union
import uuid
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi import HTTPException

"""
app.py
This module implements a FastAPI application for an e-commerce machine learning model API.
It provides endpoints for creating, loading, retrieving, and updating product items, as well as predicting product categories.
Redis is used as the backend storage for product items, and Jinja2 is used for HTML template rendering.
Endpoints:
----------
- POST /create_item/:
    Create a new item and store it in Redis.
    Parameters: sku (int), name (str), description (str), category (str, optional), predict_category (str, optional)
    Returns: Item
- PUT /load_items/:
    Load multiple items from a data source into Redis.
    Returns: dict with total items and total loaded items.
- GET /get_item/:
    Retrieve an item from Redis by SKU.
    Parameters: sku (str)
    Returns: Item
- GET /predict_category/:
    Predict the category of a product based on its name and description.
    Parameters: name (str), description (str)
    Returns: str (predicted category)
- PUT /update_predict_category/:
    Update the predicted category of a product item in Redis.
    Parameters: sku (str)
    Returns: dict (updated item data)
- GET /:
    Render an HTML form for item creation and prediction.
    Returns: HTMLResponse
- POST /new_item_and_predict/:
    Create a new item, predict its category, store it in Redis, and render the result in an HTML template.
    Parameters: name (str, form), description (str, form), category (str, form, optional)
    Returns: HTMLResponse (rendered template with item)
Classes:
--------
- Item (pydantic.BaseModel):
    Represents a product item with SKU, name, description, category, and predicted category.
Dependencies:
-------------
- FastAPI
- Pydantic
- Redis
- Jinja2Templates
- Starlette
- src.preprocessdata.read_data
- src.settings.redis_port
- src.predict.main_predict
Notes:
------
- Redis is used for storing and retrieving product items.
- The application supports both API and HTML form interactions.
- Category prediction is performed using an external ML model function.
"""

# Initialize FastAPI app and Redis client
app =  FastAPI(host="localhost", title="ecommerce", description="API for ecommerce ml model", version="1.0", docs_url="/docs")
# Initialize Redis client
redis_client = redis.StrictRedis(host='redis', port=redis_port, db=0)
# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class Item(BaseModel):
    sku: str
    name: str
    description: str
    category: Union[str, None] = None
    predict_category: Union[str, None] = None

@app.post("/create_item/")
# Create a new item and store it in Redis
async def create_item(sku:int, name:str, description:str, category:str=None, predict_category:str=None) -> Item:
    item = Item(sku=str(sku), name=str(name), description=str(description), category=category, predict_category=predict_category)
    redis_client.hmset(str(item.sku), {
        "name": item.name, "description": item.description,
        "category": str(item.category), "predict_category": str(item.predict_category)
        })
    return item

@app.put("/load_items/")
# Load multiple items from data source into Redis
async def load_items() -> dict:
    products = read_data()
    load_items = 0
    # Iterate through products and load into Redis
    for item in products.to_dict('records'):
        try:
            # Use SKU as the key and store other attributes as a hash
            redis_client.hmset(item["sku"], {
                "name": item["name"], "description": item["description"],
                "category": str(item["category"][-1]["id"]), "predict_category": str(None)
                })
            load_items = load_items +1
        except redis.exceptions.DataError:
            print("item with none type data")

    return {"total_items": len(products), "total_load_items": load_items}

@app.get("/get_item/", response_model=Item)
async def get_item(sku: str) -> Item:
    raw = redis_client.hgetall(str(sku))
    if not raw:
        raise HTTPException(status_code=404, detail="SKU not found")

    data = {k.decode(): v.decode() for k, v in raw.items()}

    def nonefix(x):
        return None if x in (None, "", "None", "null", "NULL") else x

    return Item(
        sku=str(sku),
        name=data.get("name", ""),
        description=data.get("description", ""),
        category=nonefix(data.get("category")),
        predict_category=nonefix(data.get("predict_category")),
    )

@app.get("/predict_category/")
# Predict the category of a product based on its name and description
async def predict_category(name: str, description: str) -> str:
    return main_predict(name, description)

@app.put("/update_predict_category/")
# Update the predicted category of a product item in Redis
async def update_predict_category(sku: str) -> dict:
    item = redis_client.hgetall(str(sku))
    predict_category_value = main_predict(item[b"name"], item[b"description"])
    redis_client.hmset(str(sku), {
        "name": item[b"name"], "description": item[b"description"],
            "category": item[b"category"], "predict_category": predict_category_value
            })
    return redis_client.hgetall(str(sku))

@app.get("/", response_class=HTMLResponse)
# Render an HTML form for item creation and prediction
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/new_item_and_predict/")
# Create a new item, predict its category, store it in Redis, and render the result in an HTML template
async def new_item_and_predict(request: Request, name:str = Form(...), description:str = Form(...), category:str = Form(None)) -> dict:
    import random
    import redis
    # Generate a random number between 1,000,000 and 9,999,999
    sku = random.randint(1000000, 9999999)
    # Check if the generated number already exists in Redis
    while redis_client.exists(sku):
    # If it exists, generate a new number
        sku = random.randint(1000000, 9999999)
    # Predict the category using the provided name and description
    predict_category_value = main_predict(name, description)
    item = Item(sku=str(sku), name=str(name), description=str(description), category=str(category), predict_category=str(predict_category_value))

    redis_client.hmset(str(item.sku), {
        "name": str(item.name), "description": str(item.description),
        "category": str(item.category), "predict_category": str(item.predict_category)
        })
    return templates.TemplateResponse("result.html", {"request": request, "item": item})
