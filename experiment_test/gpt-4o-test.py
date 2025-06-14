import os
import json
from tqdm import tqdm
from openai import OpenAI  # from openai-python library

# 🛠️ CONFIGURATION
INPUT_FILE = 'experiment_test/test_llm.jsonl'
OUTPUT_FOLDER = 'gpt4o_outputs'
OUTPUT_FILE = 'gpt4o_generated.jsonl'
MODEL = 'gpt-4o'  # GPT‑4o API model

# Set your API key via environment variable
# export OPENAI_API_KEY="your_api_key_here"

# Setup client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prepare output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

def format_prompt(instruction, context):
    if context.strip():
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            "### Response:"
        )
    else:
        return (
            "Below is an instruction that describes a task.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:"
        )

def generate_with_gpt4o(prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content

# Read input dataset
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    entries = [json.loads(line) for line in f]

# Batch inference
with open(output_path, 'w', encoding='utf-8') as out_f:
    for entry in tqdm(entries, desc="GPT‑4o inference"):
        instruction = entry.get("instruction", "")
        context = entry.get("input", "")
        prompt = format_prompt(instruction, context)
        try:
            generated = generate_with_gpt4o(prompt)
        except Exception as e:
            generated = f"[Error generating response: {e}]"
        result = {
            "instruction": instruction,
            "input": context,
            "generated_output": generated
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"✅ Completed GPT‑4o inference. Results saved to: {output_path}")
