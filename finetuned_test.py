import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, './lora_adapter').to(device)
# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')

def format_prompt(instruction, context):
    """Formats the prompt as it was structured during training."""
    return f"Instruction: {instruction}\nContext: {context}\nAnswer:"

def generate_response(instruction, context, device, max_length=3072):
    """Generates a response from the fine-tuned model given an instruction and context."""
    # Construct the prompt
    prompt = format_prompt(instruction, context)
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)

    # Decode the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    device = "cuda"
    while True:
        user_instruction = input("Enter your instruction (or type 'exit' to quit): ")
        if user_instruction.lower() == "exit":
            break
        user_context = input("Enter the related context: ")
        
        response = generate_response(user_instruction, user_context, device)
        print("\nGenerated Response:\n", response, "\n")
