from cs336_basics.model import BasicsTransformerLM
import torch
from torch.nn import Module


model_conf = {"vocab_size": 50257, "context_length": 1024, 
              "d_model": 768, "num_layers": 12, 
              "num_heads": 12, "d_ff": 3072}

model = BasicsTransformerLM(**model_conf)

def param_counting(model: Module) -> int:
    count = 0
    for param in model.parameters():
        count += param.numel()
    return count

print(param_counting(model))