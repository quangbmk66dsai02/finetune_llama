from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
llama_path = "meta-llama/Llama-3.2-3B-Instruct"
device = "cuda"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_path)
llama_model = AutoModelForCausalLM.from_pretrained(llama_path, torch_dtype=torch.float16)
print("Finish loading")


query="Đại tướng Võ Nguyên Giáp sinh ra ở đâu?"
answer_content = ""
prompt = (
    f"Bạn là một trợ lý hỗ trợ QA lịch sử. Hãy trả lời câu hỏi sau một cách chi tiết, "
    f"dựa trên thông tin được cung cấp. Bỏ qua những thông tin không liên quan.\n\n"
    f"Câu hỏi: {query}\n"
    f"Nội dung tham khảo:\n{answer_content}\n\n"
    f"Câu trả lời:"
)

# Tokenize input
inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = llama_model.generate(**inputs, max_length=1024, temperature=0.7, top_p=0.9)
response = llama_tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("THIS IS THE RESPONSE CONTENT", response)
