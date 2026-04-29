import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, device: torch.device | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        norm = (h.norm(2) + self.eps).sqrt()[..., None]
        return h / norm * self.weight
    
def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def experiment_compile_rmsnorm():
    x = torch.randn((4, 512, 2560), requires_grad=True)
    ln = RMSNorm(x.shape[-1])

    print('-' * 20 + 'before compiling' + '-' * 20)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y: torch.Tensor = ln(x)
        y.sum().backward()

    print('-' * 20 + 'after compiling' + '-' * 20)
    ln_compiled = torch.compile(ln)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y: torch.Tensor = ln_compiled(x)
        y.sum().backward()
        

if __name__ == '__main__':
    experiment_compile_rmsnorm()

