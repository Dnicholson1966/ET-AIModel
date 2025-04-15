from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Initialize model and tokenizer (this will download them on first run)
model_name = "google/flan-t5-xl"  # Using the larger model for better responses
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_description(input_text):
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the response with careful parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=300,          # Longer responses allowed
            min_length=50,           # Ensure somewhat detailed responses
            num_return_sequences=1,
            temperature=0.8,         # Slightly more creative
            top_p=0.92,             # High-quality responses
            do_sample=True,
            no_repeat_ngram_size=3,  # Avoid repetition
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean up the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add emoji suggestions based on content
    if "price" in input_text.lower() or "cost" in input_text.lower():
        generated_text = "💰 " + generated_text
    elif "location" in input_text.lower() or "where" in input_text.lower():
        generated_text = "📍 " + generated_text
    elif "product" in input_text.lower() or "item" in input_text.lower():
        generated_text = "🛍️ " + generated_text
    elif "help" in input_text.lower() or "assist" in input_text.lower():
        generated_text = "💁 " + generated_text
    
    return generated_text.strip()