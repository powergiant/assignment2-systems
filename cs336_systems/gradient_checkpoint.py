import torch
from torch import nn
from cs336_basics.model import RotaryEmbedding, TransformerBlock, RMSNorm

# class RMSNorm(nn.Module):
#     def __init__(self, hidden_size: int, eps: float = 1e-5, device: torch.device | None = None):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
#         self.eps = eps

#     def forward(self, h: torch.Tensor) -> torch.Tensor:
#         norm = (h.norm(2) + self.eps).sqrt()[..., None]
#         return h / norm * self.weight
    
def pack_hook_rms(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook_rms(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def experiment_compile_rmsnorm():
    x = torch.randn((4, 512, 2560), requires_grad=True)
    ln = RMSNorm(x.shape[-1])

    print('-' * 20 + 'experiment_compile_rmsnorm' + '-' * 20)
    print('-' * 20 + 'before compiling' + '-' * 20)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook_rms, unpack_hook_rms):
        y: torch.Tensor = ln(x)
        y.sum().backward()

    print('-' * 20 + 'after compiling' + '-' * 20)
    ln_compiled = torch.compile(ln)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook_rms, unpack_hook_rms):
        y: torch.Tensor = ln_compiled(x)
        y.sum().backward()

    print('\n')

# Now logs the number of bytes saved
total_size_bytes = 0
def pack_hook_block(t):
    if isinstance(t, torch.nn.Parameter): # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def reset_counting():
    global total_size_bytes
    total_size_bytes = 0

def unpack_hook_block(t):
    pass

def experiment_compile_transformer_block():
    print('-' * 20 + 'experiment_compile_transformer_block' + '-' * 20)

    # num_layers for this model is 32
    d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
    block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads, positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length))
    # Fuse as much torch.compile will allow
    block = torch.compile(block, fullgraph=True)
    x = torch.randn((4, context_length, d_model), requires_grad=True)

    reset_counting()
    
    # Run forward pass, saving for backward
    with torch.autograd.graph.saved_tensors_hooks(pack_hook_block, unpack_hook_block):
        y = block(x)

    print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes /(1024**2):.2f} MiB")

    reset_counting()

    with torch.autograd.graph.saved_tensors_hooks(pack_hook_block, unpack_hook_block):
        y = block(block(block(block(x))))

    print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes /(1024**2):.2f} MiB")

    print('\n')


        

if __name__ == '__main__':
    experiment_compile_rmsnorm()
    experiment_compile_transformer_block()

