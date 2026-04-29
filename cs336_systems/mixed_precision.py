import torch
import torch.nn as nn
from torch.optim import AdamW
from torch import autocast, Tensor
import numpy as np

def experiment_accumulation():
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)

    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

def experiment_mix_precision_toy_model():
    dtype = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(10, 100).to(device)
    model = ToyModel(100, 100).to(device=device)
    opt = torch.optim.Adam(model.parameters())

    with autocast(device_type=device, dtype=dtype):
        opt.zero_grad()
        y: torch.Tensor = model(x)
        loss = y.sum()
        loss.backward()
        opt.step()

    x, loss, y

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch
from .profiler import range_profiler, find, init_profiler, get_result

def train_step(model: BasicsTransformerLM, opt: AdamW, 
               data: tuple[Tensor, Tensor], 
               step: int, is_warm_up: bool):
    opt.zero_grad()
    inputs, targets = data
    if is_warm_up:
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        opt.step()
        print(f"step: {step}, loss: {loss:.3f}, warm up")
    else:
        with range_profiler(f"forward {step}"):
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
        with range_profiler(f"backward {step}"):    
            loss.backward()
        opt.step()
        print(f"step: {step}, loss: {loss:.3f}, " + 
              f"time_forward: {find(f"forward {step}")[1] - find(f"forward {step}")[0]:.3f}, " + 
              f"time_backward: {find(f"backward {step}")[1]-find(f"backward {step}")[0]:.3f}")

from contextlib import nullcontext

def experiment_mix_precision_LM():

    model_conf = {"vocab_size": 50257, "context_length": 1024, 
              "d_model": 768, "num_layers": 12, 
              "num_heads": 12, "d_ff": 3072}

    model = BasicsTransformerLM(**model_conf)

    opt = AdamW(model.parameters())

    n_warm_up_step = 5

    n_total_step = 100

    data_config = {'dataset_size': 1000, 'batch_size': 5, 'context_length': 512}

    dataset = np.random.randint(0, model_conf["vocab_size"], data_config['dataset_size'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    autocast_context = autocast(device_type=device, dtype=torch.float16)
    # autocast_context = autocast(device_type=device, dtype=torch.bfloat16)
    # autocast_context = nullcontext()

    model.to(device)
    init_profiler()
    for step in range(n_total_step):
        data = get_batch(dataset=dataset, 
                         batch_size=data_config['batch_size'], 
                         context_length=data_config['context_length'],
                         device=device)
        
        with autocast_context:
            train_step(model, opt, data, step, step < n_warm_up_step)

    print(get_result())

# TODO: adding a transformer training script

if __name__ == '__main__':
    # experiment_accumulation()
    # experiment_mix_precision_toy_model()
    experiment_mix_precision_LM()
    
    