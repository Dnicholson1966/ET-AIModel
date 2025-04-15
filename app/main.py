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

@app.post("/", response_class=HTMLResponse)
async def process_form(request: Request, prompt: str = Form(...)):
    # Enhance the prompt with Walmart context
    walmart_context = "You are a Walmart Store Assistant AI. Provide helpful, friendly responses "\
                     "about Walmart products, services, store information, and policies. "\
                     "Keep responses concise and focused on Walmart-related information."
    
    enhanced_prompt = f"{walmart_context}\n\nCustomer question: {prompt}"
    
    # Generate response using our AI model
    response = generate_description(enhanced_prompt)
    
    # Pass both prompt and response back to template
    return templates.TemplateResponse("WalmartKiosk.html", {
        "request": request,
        "prompt": prompt,
        "response": response
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