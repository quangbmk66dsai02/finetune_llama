from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
llama_path = "meta-llama/Llama-3.2-3B-Instruct"
device = "cuda"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_path)
llama_model = AutoModelForCausalLM.from_pretrained(llama_path, torch_dtype=torch.float16, device_map="auto")
print("Finish loading")
tokenizer = llama_tokenizer
tokenizer.pad_token = tokenizer.eos_token
model = llama_model

validation_texts = ["What is the most common type of conflict in literature?", "Why do we need sleep?", "Name one important consequence of deforestation."]
inputs = tokenizer(validation_texts, return_tensors='pt', padding=True, truncation=True).to(device)
input_length = inputs['input_ids'].shape[1]

model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=input_length + 50)  # Adjust max_length as needed

# Exclude the input prompt from the generated output
generated_tokens = outputs
predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

for pred in predictions:
    print("Generated Text:", pred)

