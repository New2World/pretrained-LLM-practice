import torch
import math
import time
import sys

from datasets import load_dataset
import evaluate
from transformers import Gemma3ForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from torch.utils.data import DataLoader

def load_chat_template(path):
    with open(path, "r") as f:
        chat_template = f.read()
    return chat_template

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
model.eval()

sst_dataset = load_dataset("glue", "sst2")["validation"]
sst_dataloader = DataLoader(sst_dataset, batch_size=1)

chat_template = load_chat_template("../gemma_openhermes_template.txt")
gt = []
pred = []
with torch.inference_mode():
    torch.cuda.empty_cache()
    system_head = [{"from": "system", "value": "Carefully analyze the sentiment and output ONLY 'positive' or 'negative' in WHOLE WORD and LOWER CASE. DO NOT OUTPUT ANY OTHER WORDS."}]
    for sample in sst_dataloader:
        chats = [{"from": "human", "value": "\""+sample["sentence"][0]+"\""}]
        inputs = tokenizer.apply_chat_template(system_head + chats,
                                               tokenize=True,
                                               chat_template=chat_template,
                                               return_tensors="pt",
                                               return_dict=True,
                                               add_generation_prompt=True)
        prompt_len = len(inputs["input_ids"][0])
        inputs = inputs.to(model.device)
        output_token = model.generate(inputs["input_ids"], 
                                      attention_mask=inputs["attention_mask"], 
                                      max_new_tokens=10, 
                                      temperature=0.01, 
                                      use_cache=False)
        output = tokenizer.decode(output_token[0][prompt_len:], skip_special_tokens=True).lower()
        if "negative" in output:
            pred.append(0)
        else:
            pred.append(1)
        gt.append(sample["label"][0].data)

accuracy = evaluate.load("accuracy")
result = accuracy.compute(predictions=pred, references=gt)
print(result)