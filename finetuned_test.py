import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, './lora_adapter').to(device)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')

def format_prompt(instruction, context):
    """Formats the prompt to match the Alpaca dataset structure."""
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

def generate_response(instruction, context, device, max_length=512):
    """Generates a response from the fine-tuned model given an instruction and context."""
    prompt = format_prompt(instruction, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        user_instruction = input("Enter your instruction (or type 'exit' to quit): ")
        if user_instruction.lower() == "exit":
            break
        user_context = input("Enter the related context: ")
        response = generate_response(user_instruction, user_context, device)
        print("\nGenerated Response:\n", response, "\n")
