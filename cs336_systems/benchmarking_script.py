from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
from torch import Tensor
from torch.nn import Module
from .profiler import TorchStepProfiler, get_result, find, init_profiler
import numpy as np

def param_counting(model: Module) -> int:
    count = 0
    for param in model.parameters():
        count += param.numel()
    return count

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
        with TorchStepProfiler(f"forward {step}"):
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
        with TorchStepProfiler(f"backward {step}"):    
            loss.backward()
        opt.step()
        print(f"step: {step}, loss: {loss:.3f}, \
              time_forward: {find(f"forward {step}")}, \
              time_backward: {find(f"backward {step}")}")


if __name__ == '__main__':
    model_conf = {"vocab_size": 50257, "context_length": 1024, 
              "d_model": 768, "num_layers": 12, 
              "num_heads": 12, "d_ff": 3072}

    model = BasicsTransformerLM(**model_conf)
    print(param_counting(model))

    opt = AdamW(model.parameters())

    n_warm_up_step = 5

    n_total_step = 100

    data_config = {'dataset_size': 100000, 'batch_size': 5, 'context_length': 512}

    dataset = np.random.randint(0, model_conf["vocab_size"], data_config['dataset_size'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    init_profiler()
    for step in range(n_total_step):
        data = get_batch(dataset=dataset, 
                         batch_size=data_config['batch_size'], 
                         context_length=data_config['context_length'],
                         device=device)
        train_step(model, opt, data, step, step < n_warm_up_step)

    print(get_result())
        

