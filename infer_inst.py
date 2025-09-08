import torch
import math
import time
import sys
from termcolor import colored

from transformers import Gemma3ForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel, PeftConfig

from myutils import load_chat_template

assert len(sys.argv) == 2
pretrained_model = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision('high')

peft_config = PeftConfig.from_pretrained(pretrained_model)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, cache_dir="/scratch/ruw400/.hf_hub")
base_model = Gemma3ForCausalLM.from_pretrained(peft_config.base_model_name_or_path, 
                                               torch_dtype=torch.bfloat16, 
                                               cache_dir="/scratch/ruw400/.hf_hub", 
                                               device_map="auto").to(device)
model = PeftModel.from_pretrained(base_model, 
                                  pretrained_model, 
                                  torch_dtype=torch.bfloat16, 
                                  cache_dir="/scratch/ruw400/.hf_hub", 
                                  device_map="auto").to(device)

chat_template = load_chat_template("gemma_openhermes_template.txt")

model.eval()
with torch.inference_mode():
    system_head = [{"from": "system", "value": "I am Gemma, a helpful personal assistant."}]
    chat_cache = DynamicCache(config=model.config)
    chats = []
    while True:
        torch.cuda.empty_cache()
        prompt = input(colored(">>> ", "green"))
        if prompt == "exit":
            break
        chats.append({"from": "human", "value": prompt})
        while True:
            inputs = tokenizer.apply_chat_template(system_head + chats,
                                                   tokenize=True,
                                                   chat_template=chat_template,
                                                   return_tensors="pt",
                                                   return_dict=True,
                                                   add_generation_prompt=True)
            prompt_len = inputs["input_ids"].shape[1]
            if prompt_len > 32_000:
                chats = chats[2:]
            else:
                break

        inputs = inputs.to(model.device)
        output_token = model.generate(inputs["input_ids"], 
                                      attention_mask=inputs["attention_mask"], 
                                      max_new_tokens=32_000, 
                                      past_key_values=chat_cache,
                                      use_cache=True)
        output = tokenizer.decode(output_token[0][prompt_len:], skip_special_tokens=True)
        chats.append({"from": "gpt", "value": output})

        print(colored("<<<", "red"), output)
        print()
