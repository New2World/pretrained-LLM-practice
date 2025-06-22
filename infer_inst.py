import torch
import math
import time
import sys

from transformers import Gemma3ForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

assert len(sys.argv) == 2
pretrained_model = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision('high')

peft_config = PeftConfig.from_pretrained(pretrained_model)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, cache_dir="/scratch/ruw400/.hf_hub")
base_model = Gemma3ForCausalLM.from_pretrained(peft_config.base_model_name_or_path, 
                                            #    torch_dtype=torch.bfloat16, 
                                               cache_dir="/scratch/ruw400/.hf_hub", 
                                               device_map="auto").to(device)
model = PeftModel.from_pretrained(base_model, 
                                  pretrained_model, 
                                #   torch_dtype=torch.bfloat16, 
                                  cache_dir="/scratch/ruw400/.hf_hub", 
                                  device_map="auto").to(device)

model.eval()
with torch.inference_mode():
    while True:
        torch.cuda.empty_cache()
        prompt = input(">>> ")
        if prompt == "exit":
            break
        prompt = "###user\n" + prompt + "\n\n###response\n"
        prompt_token = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        attn_mask = torch.ones_like(prompt_token, dtype=torch.long).to(device)
        prompt_len = prompt_token.shape[1]

        output_token = model.generate(prompt_token, 
                                      attention_mask=attn_mask, 
                                      max_new_tokens=32_000)
        output = tokenizer.decode(output_token[0][prompt_len:], skip_special_tokens=True)

        print(output)
