import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM, softmax
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
from torch import Tensor
from torch.nn import Module
from .profiler import range_profiler, get_result, find, init_profiler
import numpy as np
from einops import einsum
import math

import torch.cuda.nvtx as nvtx

def param_counting(model: Module) -> int:
    count = 0
    for param in model.parameters():
        count += param.numel()
    return count

def train_step_naive_profiler(model: BasicsTransformerLM, opt: AdamW, 
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

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:

    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")





def train_step_nvtx(model: BasicsTransformerLM, opt: AdamW, 
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
        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        opt.step()
        print(f"step: {step}, loss: {loss:.3f}, profile")

if __name__ == '__main__':
    train_step = train_step_nvtx # train_step_naive_profiler

    model_conf = {"vocab_size": 50257, "context_length": 1024, 
              "d_model": 768, "num_layers": 12, 
              "num_heads": 12, "d_ff": 3072}

    model = BasicsTransformerLM(**model_conf)
    print(param_counting(model))

    opt = AdamW(model.parameters())

    n_warm_up_step = 5

    n_total_step = 100

    data_config = {'dataset_size': 1000, 'batch_size': 5, 'context_length': 512}

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
        

