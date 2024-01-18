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

app =  FastAPI(host="localhost", title="ecommerce", description="API for ecommerce ml model", version="1.0", docs_url="/docs")
redis_client = redis.StrictRedis(host='redis', port=redis_port, db=0)
templates = Jinja2Templates(directory="templates")
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

class Item(BaseModel):
    sku: str
    name: str
    description: str
    category: Union[str, None] = None
    predict_category: Union[str, None] = None

@app.post("/create_item/")
async def create_item(sku:int, name:str, description:str, category:str=None, predict_category:str=None) -> Item:
    item = Item(sku=str(sku), name=str(name), description=str(description), category=category, predict_category=predict_category)
    redis_client.hmset(str(item.sku), {
        "name": item.name, "description": item.description,
        "category": str(item.category), "predict_category": str(item.predict_category)
        })
    return item

@app.put("/load_items/")
async def load_items() -> dict:
    products = read_data()
    load_items = 0
    for item in products.to_dict('records'):
        try:
            redis_client.hmset(item["sku"], {
                "name": item["name"], "description": item["description"],
                "category": str(item["category"][-1]["id"]), "predict_category": str(None)
                })
            load_items = load_items +1
        except redis.exceptions.DataError:
            print("item with none type data")

    return {"total_items": len(products), "total_load_items": load_items}

@app.get("/get_item/")
async def read_items(sku: str) -> dict:
    return redis_client.hgetall(str(sku))

@app.get("/predict_category/")
async def predict_category(name: str, description: str) -> str:
    return main_predict(name, description)

@app.put("/update_predict_category/")
async def update_predict_category(sku: str) -> dict:
    item = redis_client.hgetall(str(sku))
    predict_category_value = main_predict(item[b"name"], item[b"description"])
    redis_client.hmset(str(sku), {
        "name": item[b"name"], "description": item[b"description"],
            "category": item[b"category"], "predict_category": predict_category_value
            })
    return redis_client.hgetall(str(sku))

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/new_item_and_predict/")
async def new_item_and_predict(request: Request, name:str = Form(...), description:str = Form(...), category:str = Form(None)) -> dict:
    import random
    import redis
    
    # Generate a random number between 1,000,000 and 9,999,999
    sku = random.randint(1000000, 9999999)

# Check if the generated number already exists in Redis
    while redis_client.exists(sku):
    # If it exists, generate a new number
        sku = random.randint(1000000, 9999999)


    predict_category_value = main_predict(name, description)
    item = Item(sku=str(sku), name=str(name), description=str(description), category=str(category), predict_category=str(predict_category_value))

    redis_client.hmset(str(item.sku), {
        "name": str(item.name), "description": str(item.description),
        "category": str(item.category), "predict_category": str(item.predict_category)
        })

    return templates.TemplateResponse("result.html", {"request": request, "item": item})


@app.get("/get_item/")
async def read_items(sku: str) -> dict:
    return redis_client.hgetall(str(sku))