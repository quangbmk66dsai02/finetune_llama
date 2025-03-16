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
dataset = load_dataset('json', data_files='sample.json')
print(dataset['train'][0])
