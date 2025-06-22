import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from accelerate import Accelerator, DeepSpeedPlugin
from transformers import Gemma3ForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

import myutils
from myutils.data import load_OpenHermes_dataset, inst_dataset_collate_fn

import sys
import time
import math
from dataclasses import dataclass
import contextlib
import inspect
from functools import partial

#####
#
#   This script suffers from OOM on 5090
#
#####

@dataclass
class TrainConfig:
    init_lr: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 200
    max_steps: int = 200_000

    device_type: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
    grad_accum: int = 8
    grad_clip: float = 0.0

class Trainer:
    def __init__(self, config):
        self.config = config
    
    def get_lr_scheduler(self, optimizer, last_step=-1):
        lrf = self.config.min_lr / self.config.init_lr
        cosine_schd = optim.lr_scheduler.LambdaLR(optimizer, partial(myutils.lr_onecycle, n_steps=self.config.max_steps-self.config.warmup_steps, lrf=lrf), last_epoch=last_step)
        if last_step >= 0:
            return cosine_schd
        warmup_schd = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.config.warmup_steps)
        lr_schd = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_schd, cosine_schd], milestones=[self.config.warmup_steps])
        return lr_schd
    
    def get_optimizer(self, model, lr, weight_decay, device_type, **kargs):
        fused = (device_type=="cuda") and ("fused" in inspect.signature(optim.AdamW).parameters)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=fused)
        return optimizer
    
    def train(self, model, dataloader):
        global_steps = 0
        verbose_steps = 50

        deepspeed_plugin = DeepSpeedPlugin(zero_stage=1, 
                                        #    offload_optimizer_device="cpu", 
                                           gradient_clipping=None)
        accelerator = Accelerator(mixed_precision=self.config.dtype,
                                  gradient_accumulation_steps=self.config.grad_accum,
                                  deepspeed_plugin=deepspeed_plugin, 
                                  step_scheduler_with_optimizer=False)

        optimizer = self.get_optimizer(model, self.config.init_lr, self.config.weight_decay, self.config.device_type)
        optimizer.zero_grad(set_to_none=True)
        lr_schd = self.get_lr_scheduler(optimizer, global_steps-1)

        model, dataloader, optimizer, lr_schd = accelerator.prepare(model, dataloader, optimizer, lr_schd)
        while global_steps < self.config.max_steps:
            avg_loss = 0.
            for sample in dataloader:
                with accelerator.accumulate(model):
                    outputs = model(sample["input_ids"], labels=sample["labels"])
                    loss = outputs.loss
                    avg_loss += outputs.loss.item()

                    accelerator.backward(loss)

                    if self.config.grad_clip != 0.0 and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    
                    optimizer.step()
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)
                    del outputs

                    if accelerator.sync_gradients:
                        lr_schd.step()
                        global_steps += 1
                        # torch.cuda.empty_cache()

                        # log
                        if global_steps % verbose_steps == 0 and accelerator.is_main_process:
                            # lossf = accelerator.gather(avg_loss / accelerator.gradient_accumulation_steps).sum()
                            lossf = avg_loss / accelerator.gradient_accumulation_steps / verbose_steps
                            avg_loss = 0.
                            myutils.record_print(f"step {global_steps}: loss {lossf:.4f}, lr {lr_schd.get_last_lr()[0]:.6f}")

                            if global_steps % 1_000 == 0:
                                sys.stdout.flush()
                        
                        if global_steps % 10_000 == 0 and accelerator.is_main_process:
                            accelerator.unwrap_model(model).save_pretrained(f"log/model/gemma_it_{global_steps:010d}", from_pt=True)
                        
                        if global_steps > self.config.max_steps:
                            break


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True    # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True          # allow tf32 on cudnn

    ddp_world_size = torch.cuda.device_count()
    assert args.grad_accum >= 1
    assert args.grad_accum % ddp_world_size == 0
    
    from huggingface_hub import login
    login(token="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")
    model = Gemma3ForCausalLM.from_pretrained(args.pretrained, cache_dir=args.cache_dir, device_map="auto")
    if args.lora_r > 0:
        print("[ Using LoRA ]")
        lora_config = LoraConfig(r=args.lora_r,
                                lora_alpha=args.lora_alpha,
                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                lora_dropout=0.05,
                                bias="none",
                                task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, lora_config)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt", cache_dir=args.cache_dir)

    it_dataset = load_OpenHermes_dataset(tokenizer, args.cache_dir)
    dataloader = DataLoader(it_dataset,
                            batch_size=args.batch_size,
                            collate_fn=partial(inst_dataset_collate_fn, pad_token_id=tokenizer.pad_token_id),
                            shuffle=True,
                            num_workers=8)
    
    trainer_config = TrainConfig(init_lr=args.init_lr,
                                 min_lr=args.min_lr,
                                 weight_decay=args.weight_decay,
                                 warmup_steps=args.warmup_steps,
                                 max_steps=args.max_steps,
                                 grad_accum=args.grad_accum // ddp_world_size,
                                 grad_clip=args.grad_clip)
    print(f"===== Trainer config =====")
    print(trainer_config)
    print()
    trainer = Trainer(trainer_config)
    trainer.train(model, dataloader)

if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Gemma Fine-Tuning")
    parser.add_argument("--max_steps", type=int, default=200_000, help="max steps of training")      # tokens = max_steps * (block_size * batch_size * grad_accum)
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="accumulate gradient")
    parser.add_argument("--init_lr", type=float, default=1e-5, help="initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="minimal learning rate")
    parser.add_argument("--grad_checkpointing", action="store_true", default=False, help="enable gradient checkpointing")
    parser.add_argument("--lora_r", type=int, default=0, help="LoRA rank, default `0` for disabling LoRA")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--warmup_steps", type=int, default=200, help="warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="optimizer weight decay")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="gradient clipping")
    parser.add_argument("--pretrained", type=str, default=None, help="load pretrained model")
    parser.add_argument("--cache_dir", type=str, default=None, help="cache path")

    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
    args = parser.parse_args()

    if args.cache_dir and os.path.isdir(args.cache_dir):
        os.environ["HF_HOME"] = args.cache_dir

    main(args)