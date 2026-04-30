import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elem, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    print('pid', pid)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elem

    x = tl.load(x_ptr + offsets, mask)
    y = tl.load(y_ptr + offsets, mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask)


device = 'cuda'
n_elem = 4096
x = torch.rand(n_elem, device=device)
y = torch.rand(n_elem, device=device)
output = torch.empty_like(x)
grid = lambda meta: (triton.cdiv(n_elem, meta['BLOCK_SIZE']), )
add_kernel[grid](x, y, output, n_elem, BLOCK_SIZE=1024)