import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = 'meta-llama/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
tokenizer.pad_token = tokenizer.eos_token

lora_model_path = './fine_tuned_llama3_2_1b_instruct'
model = PeftModel.from_pretrained(model, lora_model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


validation_texts = ["Name one important consequence of deforestation."]
inputs = tokenizer(validation_texts, return_tensors='pt', padding=True, truncation=True).to(device)
input_length = inputs['input_ids'].shape[1]

model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=input_length + 50)  # Adjust max_length as needed

# Exclude the input prompt from the generated output
generated_tokens = outputs[:, input_length:]
predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

for pred in predictions:
    print("Generated Text:", pred)


for pre in predictions:
    print("THIS IS PRE", pre)