import torch

import os
import math
from functools import wraps, partial
import datetime


def timestamp(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(f'{datetime.datetime.now().strftime("%m-%d %H:%M:%S"):16}|', *args, **kwargs)
    return wrapper

debug_print = partial(timestamp(print), flush=True)
record_print = partial(timestamp(print), flush=False)


def load_optimizer(path, optimizer, device_map=None):
    if os.path.exists(path):
        optim_sd = torch.load(path, map_location=device_map)
        optimizer.load_state_dict(optim_sd)
    return optimizer

def save_optimizer(path, optimizer):
    torch.save(optimizer.state_dict(), path)

def lr_linear(x, n_steps, lrf):
    return (1 - x/n_steps) * (1. - lrf) + lrf

def lr_onecycle(x, n_steps, lrf):
    return ((1 - math.cos(x * math.pi / n_steps)) / 2) * (lrf - 1) + 1
