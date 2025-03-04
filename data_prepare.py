from datasets import load_dataset

# Load the Alpaca dataset
dataset = load_dataset('tatsu-lab/alpaca')

import random

# Calculate 1% of the dataset size
subset_size = int(0.002 * len(dataset['train']))

# Set a seed for reproducibility
random.seed(42)

# Randomly select indices for the subset
subset_indices = random.sample(range(len(dataset['train'])), subset_size)

# Create the subset
subset = dataset['train'].select(subset_indices)

import json

formatted_data = []

for entry in subset:
    formatted_entry = {
        'instruction': entry['instruction'],
        'input': entry.get('input', ''),
        'output': entry['output']
    }
    formatted_data.append(formatted_entry)

# Save the formatted subset to a JSON file
with open('alpaca_subset.json', 'w') as f:
    json.dump(formatted_data, f, indent=4)
