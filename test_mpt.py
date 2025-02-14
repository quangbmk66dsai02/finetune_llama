import torch
import transformers

name = 'mosaicml/mpt-7b-storywriter'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


from transformers import pipeline

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

with torch.autocast('cuda', dtype=torch.bfloat16):
    print(
        pipe('Here is a recipe for vegan banana bread:\n',
            max_new_tokens=1000,
            do_sample=True,
            use_cache=True))


