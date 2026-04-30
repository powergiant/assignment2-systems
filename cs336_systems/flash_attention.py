import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output, n_elem, BLOCK_SIZE):
    print('pid', tl.program_id(0))

n_elem = 4096
x = torch.rand(n_elem)
y = torch.rand(n_elem)
output = torch.empty_like(x)
grid = lambda meta: (triton.cdiv(n_elem, meta['BLOCK_SIZE']), )
add_kernel[grid](x, y, output, n_elem, BLOCK_SIZE=1024)