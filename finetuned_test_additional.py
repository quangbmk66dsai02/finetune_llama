from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from datasets import load_dataset

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-3B-Instruct',
    torch_dtype=torch.float16,
    device_map="auto"
)
<<<<<<< HEAD
dataset = load_dataset('json', data_files='data/test_data.json')['train']
=======
dataset = load_dataset('json', data_files='data/combined-5k-line.jsonl')['train']
>>>>>>> 9b1dd37c8e1e32f92a6ca5786fabfb49702cd4dd

# Load the first adapter
model = PeftModel.from_pretrained(base_model, './lora-adapter')
model = PeftModel.from_pretrained(model, './lora-adapter2').to("cuda")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')





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

def generate_response(instruction, context, device, max_length=1512):
    prompt = format_prompt(instruction, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length + prompt_len,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][prompt_len:]  # Skip the prompt
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        try:
            user_input = input("Enter the dataset entry number (or type 'exit' to quit): ")
            if user_input.lower() == "exit":
                break
            entry_number = int(user_input)
            if 0 <= entry_number < len(dataset):
                entry = dataset[entry_number]
                instruction = entry.get('instruction', '')
                context = entry.get('input', '')
                print(f"\nInstruction:\n{instruction}\n")
                print(f"Context:\n{context}\n")
                response = generate_response(instruction, context, device="cuda")
                print("\nGenerated Response:\n", response, "\n")
            else:
                print(f"Please enter a number between 0 and {len(dataset) - 1}.")
        except Exception as e:
            print(f"\nAn error occurred: {e}\nContinuing to the next iteration...\n")
