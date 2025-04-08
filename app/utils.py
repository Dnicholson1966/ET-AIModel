from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Initialize model and tokenizer (this will download them on first run)
model_name = "google/flan-t5-large"  # A good general-purpose model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_description(input_text):
    # Prepare the prompt
    prompt = """Generate a product description with emojis from the following information:
    
    Information: {input}
    
    Description:""".format(input=input_text)
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.strip()