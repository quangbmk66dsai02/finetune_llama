from transformers import AutoTokenizer

# Load the Llama 3.2 tokenizer
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
tokenizer = AutoTokenizer.from_pretrained("vilm/vinallama-7b-chat")

while True:
    token_id_tmp = input("Please enter tokens \n").split(",")
    # Example token IDs (you should have a list of token IDs from encoding)
    token_ids = [int(id) for id in token_id_tmp]
    decoded_input = tokenizer.decode(token_ids)
    print("=====================================================================")
    print(decoded_input)