import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from datasets import load_dataset
from tqdm import tqdm

# CONFIG
INPUT_FILE = 'data/test_data.json'             # Input JSONL file
OUTPUT_FOLDER = 'inference_outputs'            # Folder to save results
OUTPUT_FILE = 'generated_responses.jsonl'      # Output JSONL file name
MAX_LENGTH = 1512                              # Max new tokens

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

# Load dataset
dataset = load_dataset('json', data_files=INPUT_FILE)['train']

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-3B-Instruct',
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, './lora-adapter')
model = PeftModel.from_pretrained(model, './lora-adapter2').to("cuda")
model.eval()

def format_prompt(instruction, context):
    if context.strip():
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            "### Response:"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:"
        )

def generate_response(instruction, context, device="cuda"):
    prompt = format_prompt(instruction, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH + prompt_len,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

# Process each entry and generate response
results = []
for entry in tqdm(dataset, desc="Generating responses"):
    instruction = entry.get('instruction', '')
    context = entry.get('input', '')
    output = generate_response(instruction, context)
    results.append({
        'instruction': instruction,
        'input': context,
        'generated_output': output
    })

# Save results to output file
with open(output_path, 'w', encoding='utf-8') as f:
    for item in results:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"\n✅ Inference complete. Results saved to: {output_path}")
