from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("./lora-llama3.2-finance")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    "./lora-llama3.2-finance",
    torch_dtype=torch.float16,
    device_map="auto",
)
model.to(device)

data_prompt = """Analyze the provided text from a financial perspective.

### Input:
{}

### Response:
{}"""
# Define the input prompt
prompt = """Pay off car loan entirely or leave $1 until the end of the loan period?
"""
finished_data_prompt = data_prompt.format(prompt,
                   "",)
# Tokenize the input prompt
inputs = tokenizer(finished_data_prompt, return_tensors="pt").to(device)

print("THIS IS THE PROMPT", finished_data_prompt)
# Tokenize the prompt
# Generate the response
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=500)

# Decode the generated tokens to text
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract the response part
response = response.split("Response:")[-1].strip()

print(response)

