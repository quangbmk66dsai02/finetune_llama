import torch
from transformers import pipeline

model_id = "ngoan/Llama-2-7b-vietnamese-20k"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
while True:
    input_text = input("Input: ")
    if input_text == "exit":
        break
    print(pipe(input_text))
