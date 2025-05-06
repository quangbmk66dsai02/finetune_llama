import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
import torch
import time

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
            sample = self.dataset[self.step_count % len(self.dataset)]
            instruction = sample['instruction']
            input_text = sample['input']

            if input_text.strip():
                prompt = (
                    "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n"
                    "### Response:"
                )
            else:
                prompt = (
                    "Below is an instruction that describes a task.\n\n"
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    "### Response:"
                )

            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = kwargs['model'].generate(
                    **inputs,
                    max_new_tokens=1000,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"\n\nStep {self.step_count} {'='*100}\nGenerated text:\n{generated_text}\n{'='*100} END")


# Load the dataset
dataset = load_dataset('json', data_files='data/vi-alpaca.json')
# subset_dataset = dataset['train'].select(range(50))
subset_dataset = dataset['train'] #Take full

print(dataset)
filtered_dataset = dataset['train'].filter(lambda example: len(example['output']) <= 512)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')

tokenizer.pad_token = tokenizer.eos_token

# Define the tokenization function
def tokenize_function(examples):
    inputs = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        if input_text.strip():
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. \n\n"
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. \n\n"
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )
        inputs.append(prompt)

    model_inputs = tokenizer(
        inputs,
        padding='max_length',
        truncation=True,
        max_length=3024,
        return_tensors='pt'
    )
    model_inputs['labels'] = model_inputs['input_ids'].clone()
    return model_inputs


# Tokenize the dataset
tokenized_dataset = subset_dataset.map(tokenize_function, batched=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-3B-Instruct',
    torch_dtype=torch.float16
)
# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# Apply the LoRA configuration to the model
model = get_peft_model(model, lora_config)
model.to(device)
input("Finished loading model, press Enter to continue...")
# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
)

generate_text_callback = GenerateTextCallback(tokenizer, subset_dataset, device, n_steps=50000)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    callbacks=[generate_text_callback]
)

start_time = time.time()

# Start the training process
trainer.train()

end_time = time.time()
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")

# Save the fine-tuned model and tokenizer
model.save_pretrained('./lora-adapter')
