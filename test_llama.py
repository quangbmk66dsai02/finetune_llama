import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Move the model to the GPU
model.to(device)

data_prompt = """Analyze the provided text from a financial perspective.

### Input:
{}

### Response:
{}"""
# Define the input prompt
prompt = """Pay off car loan entirely or leave $1 until the end of the loan period?
"""
finished_data_prompt = data_prompt.format(prompt, "")

# Tokenize the input prompt and move input tensors to the GPU
inputs = tokenizer(finished_data_prompt, return_tensors="pt").to(device)

print("THIS IS THE PROMPT", finished_data_prompt)

# Generate a response
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=500)

# Decode and print the generated text
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
