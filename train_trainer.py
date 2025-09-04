import torch
from trl import SFTTrainer
from transformers import (
    Trainer, TrainingArguments,
    Gemma3ForCausalLM, AutoTokenizer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import myutils
from myutils.data import load_OpenHermes_dataset_chat_template, inst_dataset_collate_fn
from functools import partial
import os
import argparse
import math
from transformers.trainer_pt_utils import get_parameter_names
from torch.optim import AdamW
from accelerate import Accelerator


def get_onecycle_scheduler(optimizer, num_training_steps, min_lr_ratio, warmup_steps):
    def lr_lambda(x):
        return ((1 - math.cos(x * math.pi / (num_training_steps-warmup_steps))) / 2) * (min_lr_ratio - 1) + 1
    cosine_schd = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    warmup_schd = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    lr_schd = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_schd, cosine_schd], milestones=[warmup_steps])
    return lr_schd


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    accelerator = Accelerator(
        mixed_precision=args.dtype,
        gradient_accumulation_steps=args.grad_accum,
    )

    myutils.huggingface_login()
    model = Gemma3ForCausalLM.from_pretrained(
        args.pretrained,
        cache_dir=args.cache_dir,
        device_map="auto",
        attn_implementation='eager'
    )

    ## Solve the "no grad_fn" problem
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    ##

    if args.lora_r > 0:
        print("[ Using LoRA ]")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)

    if args.grad_checkpointing:
        print("[ Gradient checkpointing enabled ]")
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt", cache_dir=args.cache_dir)

    dataset = load_OpenHermes_dataset_chat_template(tokenizer, args.cache_dir, split="train")
    if not isinstance(dataset, Dataset):
        dataset = Dataset.from_list(dataset)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.init_lr)


    custom_scheduler = get_onecycle_scheduler(
        optimizer=optimizer,
        num_training_steps=args.max_steps,
        min_lr_ratio=args.min_lr / args.init_lr,
        warmup_steps=args.warmup_steps
    )

    model, optimizer, dataset = accelerator.prepare(model, optimizer, dataset)

    training_args = TrainingArguments(
        output_dir="log/model",
        label_names=["labels"],
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.init_lr,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_steps=10_000,
        save_total_limit=5,
        fp16=(args.dtype == "fp16"),
        bf16=(args.dtype == "bf16"),
        gradient_checkpointing=args.grad_checkpointing,
        ddp_find_unused_parameters=False,
        deepspeed="ds_config_zero1.json",
        logging_dir="log/tensorboard",
        report_to=["tensorboard"],
        lr_scheduler_type="constant_with_warmup",
        torch_empty_cache_steps=10_000,
        disable_tqdm=True
    )

    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # tokenizer=tokenizer,
        data_collator=default_data_collator,
        optimizers=(optimizer, custom_scheduler),
    )

    trainer.train()

    # trainer.save_model("log/model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma Fine-Tuning with HuggingFace Trainer")
    parser.add_argument("--max_steps", type=int, default=125_200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--init_lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--grad_checkpointing", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    args = parser.parse_args()

    if args.cache_dir and os.path.isdir(args.cache_dir):
        os.environ["HF_HOME"] = args.cache_dir

    main(args)
