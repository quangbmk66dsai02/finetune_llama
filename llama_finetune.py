import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
import torch
import time  # Import the time module to track training duration


class GenerateTextCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, device, n_steps=50):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.n_steps = n_steps
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % self.n_steps == 0:
            # Select a sample from the training dataset
            sample = self.dataset[self.step_count % len(self.dataset)]
            prompt = sample['instruction']
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = kwargs['model'].generate(**inputs, max_length=500)
            
            # Decode and print the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n\nStep {self.step_count} - Prompt: {prompt}\nGenerated text:\n{generated_text}\n")


# Load the dataset
dataset = load_dataset('json', data_files='vi-alpaca.json')
filtered_dataset = dataset.filter(lambda example: len(example['output']) <= 512)
print(filtered_dataset)
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
tokenizer.pad_token = tokenizer.eos_token

# Display a sample from the dataset
print(filtered_dataset['train'][0])

# Define the tokenization function
def tokenize_function(examples):
    inputs = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        if input_text.strip():  # Check if 'input' is non-empty
            # Format with input
            inputs.append(f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}")
        else:
            # Format without input
            inputs.append(f"Instruction: {instruction}\nOutput: {output}")
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        padding='max_length',
        truncation=True,
        max_length=1536,
        return_tensors='pt'
    )
    # Set labels to be the same as input_ids
    model_inputs['labels'] = model_inputs['input_ids'].clone()
    return model_inputs


# Tokenize the dataset
tokenized_dataset = filtered_dataset.map(tokenize_function, batched=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', torch_dtype=torch.float16)


# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # Set the task type
    inference_mode=False,           # Enable training mode
    r=16,                            # Rank of the decomposition
    lora_alpha=32,                  # Scaling factor
    lora_dropout=0.1                # Dropout rate
)
# input("finished before lora")
# Apply the LoRA configuration to the model
model = get_peft_model(model, lora_config)
model.to(device)
input("finished after lora")
# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    evaluation_strategy='steps',
    eval_steps=2000000,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
)
generate_text_callback = GenerateTextCallback(tokenizer, filtered_dataset['train'], device, n_steps=1000)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['train'],  # Use a validation split if available
    callbacks=[generate_text_callback]  # Add the callback here
)

start_time = time.time()

# Start the training process
trainer.train()

end_time = time.time()
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")

# Save the fine-tuned model and tokenizer
model.save_pretrained('./lora_adapter')
