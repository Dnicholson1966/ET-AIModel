from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Initialize model and tokenizer (this will download them on first run)
model_name = "google/flan-t5-xl"  # Using FLAN-T5-XL which is freely available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision to save memory
    device_map="auto"  # Automatically handle device placement
)

# Define Walmart-specific context
WALMART_CONTEXT = """
You are a Walmart Store Assistant AI with expertise in:
1. Products & Inventory: Wide range of products including groceries, electronics, clothing, home goods
2. Services: Pharmacy, Auto Care, Vision Center, Money Services, Photo Center
3. Policies: Everyday Low Prices, Price Match Guarantee, 90-day return policy
4. Programs: Walmart+, Pickup & Delivery, Mobile Check-in
5. Store Operations: Store hours typically 6 AM-11 PM, self-checkout available

Provide friendly, accurate assistance while:
- Emphasizing value and savings
- Suggesting relevant Walmart services
- Mentioning nearby store availability when relevant
- Highlighting Walmart+ benefits where applicable
"""

def clean_response(text: str) -> str:
    """Clean and format the response text."""
    # Remove common artifacts
    text = text.replace('Assistant:', '').replace('Walmart Assistant:', '')
    text = text.strip()
    
    # Ensure proper sentence capitalization
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    # Ensure proper ending punctuation
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text

def generate_description(input_text):
    # Prepare a more focused prompt for FLAN-T5
    prompt = f"""Context: {WALMART_CONTEXT}

Question: {input_text}

Provide a concise, helpful response focusing on Walmart's offerings and services. Include specific details when possible:"""
    
    # Tokenize input with careful length handling
    inputs = tokenizer(prompt, 
                      return_tensors="pt", 
                      max_length=512,
                      truncation=True,
                      padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with carefully tuned parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,          # Shorter, more focused responses
            min_length=30,           # Ensure meaningful but concise responses
            num_return_sequences=1,
            temperature=0.8,         # Slightly more creative
            top_p=0.92,             # Higher quality threshold
            top_k=50,               # Limit vocabulary diversity
            do_sample=True,
            no_repeat_ngram_size=3,  # Avoid repetition
            length_penalty=0.8,      # Prefer slightly shorter responses
            early_stopping=True,
            bad_words_ids=[[tokenizer.encode(word)[0]] for word in ['I', 'AI', 'model', 'language']],  # Avoid self-reference
            repetition_penalty=1.3    # Stronger repetition avoidance
        )
    
    # Decode and clean up the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean the response
    response = clean_response(generated_text)
    
    # Add emoji and style based on content type
    input_lower = input_text.lower()
    if any(word in input_lower for word in ["price", "cost", "deal", "sale", "discount"]):
        response = "💰 " + response + "\n\n💡 Pro tip: Download the Walmart app for exclusive deals!"
    elif any(word in input_lower for word in ["location", "where", "store", "hours"]):
        response = "📍 " + response + "\n\n🌟 Use our store finder in the Walmart app for real-time updates."
    elif any(word in input_lower for word in ["product", "item", "stock", "inventory"]):
        response = "🛍️ " + response + "\n\n✨ Walmart+ members get early access to special items!"
    elif any(word in input_lower for word in ["help", "assist", "support", "service"]):
        response = "💁 " + response + "\n\n📱 Need more help? Try our 24/7 customer service!"
    elif any(word in input_lower for word in ["pharmacy", "prescription", "medicine"]):
        response = "💊 " + response + "\n\n⚕️ Download our app to manage prescriptions easily!"
    elif any(word in input_lower for word in ["delivery", "shipping", "pickup"]):
        response = "🚚 " + response + "\n\n🎯 Try Walmart+ for unlimited free delivery!"
    else:
        response = "🌟 " + response + "\n\n💫 We're here to help with all your shopping needs!"
    
    return response