import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

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


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
tokenizer.pad_token = tokenizer.eos_token


# Define the tokenization function


def tokenize_function(examples):
    inputs = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        if input_text.strip():
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )
        inputs.append(prompt)

    model_inputs = tokenizer(
        inputs,
        padding='max_length',
        truncation=True,
        max_length=1536,
        return_tensors='pt'
    )
    model_inputs['labels'] = model_inputs['input_ids'].clone()
    return model_inputs

# Load the new dataset

new_dataset = load_dataset('json', data_files='data/combined-5k-line.jsonl')
filtered_new_dataset = new_dataset.filter(lambda example: len(example['output']) <= 512)  #Take all
print(filtered_new_dataset)
print(filtered_new_dataset['train'][0])

# Tokenize the new dataset
tokenized_new_dataset = filtered_new_dataset.map(tokenize_function, batched=True)

# Load the already fine-tuned model
device = "cuda" if torch.cuda.is_available() else "cpu"
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,   # Set the task type
#     inference_mode=False,           # Enable training mode
#     r=16,                            # Rank of the decomposition
#     lora_alpha=32,                  # Scaling factor
#     lora_dropout=0.1                # Dropout rate
# )
base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', torch_dtype=torch.float16)
print("FINISHED LOADING BASE MODEL")
model = PeftModel.from_pretrained(base_model, 'lora-adapter', is_trainable=True).to(device)
print("FINISHED LOADING LORA MODEL")

# Apply the LoRA configuration again (if needed)
model.to(device)

generate_text_callback = GenerateTextCallback(tokenizer, filtered_new_dataset['train'], device, n_steps=1000000)
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
)

# Update the Trainer with the new dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_new_dataset['train'],
    eval_dataset=tokenized_new_dataset['train'],  # Use a validation split if available
    callbacks=[generate_text_callback]  # Keep the callback
)

# Continue training with the new dataset
start_time = time.time()

trainer.train()

end_time = time.time()
total_training_time = end_time - start_time
print(f"Total training time with new dataset: {total_training_time:.2f} seconds")

# Save the updated fine-tuned model and tokenizer
model.save_pretrained('lora-adapter-updated')