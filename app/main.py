from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.middleware.processPrompt import promptResponse
from app.middleware.modeltest import model_test
from pydantic import BaseModel
from app.utils import generate_description

app = FastAPI()

class Order(BaseModel):
    product: str
    units: int

class Product(BaseModel):
    name: str
    notes: str

class ProductDetails(BaseModel):
    name: str
    category: str
    features: list[str]
    target_audience: str

# Setup templates directory
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("WalmartKiosk.html", {"request": request})

from enum import Enum
from typing import Optional

class QueryType(Enum):
    PRODUCT = "product"
    PRICE = "price"
    LOCATION = "location"
    SERVICE = "service"
    POLICY = "policy"
    GENERAL = "general"

def detect_query_type(prompt: str) -> QueryType:
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ["cost", "price", "discount", "deal", "save"]):
        return QueryType.PRICE
    elif any(word in prompt_lower for word in ["where", "location", "nearest", "find", "store"]):
        return QueryType.LOCATION
    elif any(word in prompt_lower for word in ["service", "pharmacy", "auto", "vision", "photo"]):
        return QueryType.SERVICE
    elif any(word in prompt_lower for word in ["policy", "return", "exchange", "warranty", "guarantee"]):
        return QueryType.POLICY
    elif any(word in prompt_lower for word in ["product", "item", "sell", "stock", "inventory"]):
        return QueryType.PRODUCT
    return QueryType.GENERAL

@app.post("/", response_class=HTMLResponse)
async def process_form(request: Request, prompt: str = Form(...)):
    # Detect the type of query
    query_type = detect_query_type(prompt)
    
    # Add specific context based on query type
    query_contexts = {
        QueryType.PRODUCT: "Focus on product features, availability, and alternatives. Mention Walmart+ benefits for faster shopping.",
        QueryType.PRICE: "Emphasize Everyday Low Prices, price matching, rollbacks, and current deals. Suggest Walmart+ for extra savings.",
        QueryType.LOCATION: "Provide store hours, suggest using the Walmart app for exact locations and inventory, mention curbside pickup.",
        QueryType.SERVICE: "Detail service hours, appointment scheduling if needed, and related Walmart+ benefits.",
        QueryType.POLICY: "Explain policies clearly, mention exceptions, and suggest speaking with a store manager for special cases.",
        QueryType.GENERAL: "Provide helpful general information while highlighting relevant Walmart services and benefits."
    }
    
    # Generate response using our AI model with enhanced context
    response = generate_description(f"{prompt}\n\nContext: {query_contexts[query_type]}")
    
    # Pass both prompt and response back to template
    return templates.TemplateResponse("WalmartKiosk.html", {
        "request": request,
        "prompt": prompt,
        "response": response,
        "query_type": query_type.value  # Pass query type to template for potential UI customization
    })

## Start of API endpoints (testers)
@app.get("/ok")
async def ok_endpoint():
    return {"message": "ok"}

@app.get("/hello")
async def hello_endpoint(name: str = 'World'):
    return {"message": f"Hello, {name}!"}

@app.post("/orders")
async def place_order(product: str, units: int):
    return {"message": f"Order for {units} units of {product} placed successfully."}

@app.post("/orders_pydantic")
async def place_order(order: Order):
    return {"message": f"Order for {order.units} units of {order.product} placed successfully."}

@app.post("/product_description")
async def generate_product_description(product: ProductDetails):
    # Create a prompt from the product details
    prompt = f"Product: {product.name}\nCategory: {product.category}\nFeatures: {', '.join(product.features)}\nTarget Audience: {product.target_audience}"
    
    # Generate the description using the existing utility function
    description = generate_description(prompt)
    
    return {"description": description}