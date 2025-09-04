import torch
from datasets import load_dataset
from . import load_chat_template

from functools import partial



def OpenHermes_preprocess_fn(examples, tokenizer):
    prompt_keyword = ["system", "human"]
    content = examples["conversations"]
    prompts = "###user\n"
    responses = tokenizer.eos_token
    pointer = 0
    while content[pointer]["from"] in prompt_keyword:
        prompts += content[pointer]["value"] + "\n\n"
        pointer += 1
    prompts += "###response\n"
    responses = content[pointer]["value"] + tokenizer.eos_token

    prompt_token = tokenizer.encode(prompts, return_tensors="pt")
    response_token = tokenizer.encode(responses, return_tensors="pt")
    prompt_mask = torch.ones_like(prompt_token, dtype=torch.bool)
    response_mask = torch.zeros_like(response_token, dtype=torch.bool)
    inputs = {"input_ids": torch.concat([prompt_token, response_token], dim=1),
              "prompt_mask": torch.concat([prompt_mask, response_mask], dim=1),
              "attention_mask": torch.ones((1, prompt_token.size(-1)+response_token.size(-1)), dtype=torch.long)}
    return inputs

def load_OpenHermes_dataset(tokenizer, cache_dir):
    hermes_dataset = load_dataset("teknium/OpenHermes-2.5", cache_dir=cache_dir)
    tokenized_dataset = hermes_dataset.map(partial(OpenHermes_preprocess_fn, tokenizer=tokenizer), batched=False)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "prompt_mask"])
    return tokenized_dataset["train"]


def inst_dataset_collate_fn(batch, pad_token_id, ignore_prompt_loss=True, shift=False):
    max_length = max(x["input_ids"].size(-1) for x in batch)
    batch_size = len(batch)
    offset = 1 if shift else 0

    input_ids, labels, attention_mask = [], [], []

    ## ensure that all sequences in the same batch are padded to the same length
    for i in range(batch_size):
        l = batch[i]["input_ids"].size(-1)
        inp = torch.concat([batch[i]["input_ids"], torch.full((1, max_length-l), pad_token_id)], dim=-1)
        # cross entropy loss in PyTorch has param `ignore_index=-100` which will ignore the loss of token with value -100, avoiding meaningless computation
        lbl = torch.concat([batch[i]["input_ids"][...,offset:], torch.full((1, max_length-l+offset), -100)], dim=-1)
        att = torch.concat([batch[i]["attention_mask"], torch.zeros((1, max_length-l))], dim=-1)
        if ignore_prompt_loss:
            mask = torch.concat([batch[i]["prompt_mask"][...,offset:], torch.zeros((1, max_length-l+offset))], dim=-1).to(dtype=torch.bool)
            lbl[mask] = -100
        input_ids.append(inp)
        labels.append(lbl)
        attention_mask.append(att)

    return {"input_ids": torch.concat(input_ids),  
            "labels": torch.concat(labels), 
            "attention_mask": torch.concat(attention_mask)}


def OpenHermes_preprocess_fn_batchone(examples, tokenizer):
    prompt_keyword = ["system", "human"]
    content = examples["conversations"]
    prompts = "###user\n"
    responses = tokenizer.eos_token
    pointer = 0
    while content[pointer]["from"] in prompt_keyword:
        prompts += content[pointer]["value"] + "\n\n"
        pointer += 1
    prompts += "###response\n"
    responses = content[pointer]["value"] + tokenizer.eos_token

    prompt_token = tokenizer.encode(prompts, return_tensors="pt").squeeze()
    response_token = tokenizer.encode(responses, return_tensors="pt").squeeze()
    prompt_length = prompt_token.size(-1)
    inputs = {"input_ids": torch.concat([prompt_token, response_token]),
              "labels": torch.concat([torch.full_like(prompt_token, -100), response_token]),
              "attention_mask": torch.ones((prompt_token.size(-1)+response_token.size(-1),), dtype=torch.long)}
    return inputs

def load_OpenHermes_dataset_batchone(tokenizer, cache_dir):
    hermes_dataset = load_dataset("teknium/OpenHermes-2.5", cache_dir=cache_dir)
    tokenized_dataset = hermes_dataset.map(partial(OpenHermes_preprocess_fn_batchone, tokenizer=tokenizer), batched=False)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset["train"]


def OpenHermes_preprocess_fn_chat_template(examples, tokenizer, chat_template):
    chats = tokenizer.apply_chat_template(examples["conversations"],
                                          chat_template=chat_template,
                                          return_tensors="pt",
                                          return_dict=True,
                                          return_assistant_tokens_mask=True,
                                          add_generation_prompt=False)
    labels = chats["input_ids"].clone()
    labels[(1-chats["assistant_masks"]).to(dtype=torch.bool)] = -100
    return {"input_ids": chats["input_ids"].squeeze(),
            "attention_mask": chats["attention_mask"].squeeze(),
            "labels": labels.squeeze()}

def load_OpenHermes_dataset_chat_template(tokenizer, cache_dir, split="all"):
    assert split in ["train", "test", "all"], "split should be ['train', 'test', 'all']."
    hermes_dataset = load_dataset("teknium/OpenHermes-2.5", cache_dir=cache_dir)
    chat_template = load_chat_template("gemma_openhermes_template.txt")
    tokenized_dataset = hermes_dataset.map(partial(OpenHermes_preprocess_fn_chat_template, tokenizer=tokenizer, chat_template=chat_template), batched=False)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    if split == "all":
        return tokenized_dataset["train"]
    else:
        split_dataset = tokenized_dataset["train"].train_test_split(train_size=0.8, shuffle=False)
        return split_dataset[split]