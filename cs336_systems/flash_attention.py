import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elem, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # print('pid', pid)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elem

    x = tl.load(x_ptr + offsets, mask)
    y = tl.load(y_ptr + offsets, mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elem = output.numel()
    grid = lambda meta: (triton.cdiv(n_elem, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elem, BLOCK_SIZE=1024)
    return output

def benchmark():
    pass


def test_add_triton():
    device = 'cuda'
    n_elem = 4096
    x = torch.rand(n_elem, device=device)
    y = torch.rand(n_elem, device=device)
    print(add(x, y))

@triton.jit
def weighted_sum_forward(x_ptr, w_ptr, out_ptr,
                         x_stride_row, x_stride_dim, w_stride_dim, out_stride_row,
                         num_row, d,
                         ROW_BLOCK_SIZE: tl.constexpr, D_BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x_ptr = tl.make_block_ptr(x_ptr, shape=(num_row, d), 
                              strides=(x_stride_row, x_stride_dim),
                              offsets=pid*x_stride_row, 
                              block_shape=(ROW_BLOCK_SIZE, D_BLOCK_SIZE),
                              order=(1, 0))
    w_ptr = tl.make_block_ptr(w_ptr, shape=(d, ), 
                              strides=(w_stride_dim,),
                              offsets=(0,), 
                              block_shape=(D_BLOCK_SIZE,),
                              order=(0,))
    out_ptr = tl.make_block_ptr(out_ptr, shape=(num_row,), 
                                strides=(out_stride_row,),
                                offsets=pid*x_stride_row, 
                                block_shape=(ROW_BLOCK_SIZE,),
                                order=(0,))
    
    output = tl.zeros((ROW_BLOCK_SIZE,), dtype=tl.float32)

    for _ in range(0, d, D_BLOCK_SIZE):
        x_block = tl.load(x_ptr, boundary_check=(0, 1), padding_option='zero')
        w_block = tl.load(w_ptr, boundary_check=(0,), padding_option='zero')
        output += tl.sum(x_block * w_block[None, :], 1)

        x_block = tl.advance(x_block, (0, D_BLOCK_SIZE))
        w_block = tl.advance(w_block, (D_BLOCK_SIZE,))

    tl.store(out_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(x_ptr, w_ptr, grad_out_ptr,
                          grad_x_ptr, grad_w_partial_reduce_ptr,
                          x_stride_row, x_stride_dim, 
                          w_stride_dim, grad_out_stride_row,
                          grad_x_stride_row, grad_x_stride_dim,
                          grad_w_stride_row, grad_w_stride_dim,
                          num_row, d,
                          ROW_BLOCK_SIZE: tl.constexpr, D_BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_row_block = tl.num_programs(0)
    x_ptr = tl.make_block_ptr(x_ptr, shape=(num_row, d), 
                              strides=(x_stride_row, x_stride_dim),
                              offsets=(pid*ROW_BLOCK_SIZE, 0), 
                              block_shape=(ROW_BLOCK_SIZE, D_BLOCK_SIZE),
                              order=(1, 0))
    w_ptr = tl.make_block_ptr(w_ptr, shape=(d, ), 
                              strides=(w_stride_dim,),
                              offsets=(0,), 
                              block_shape=(D_BLOCK_SIZE,),
                              order=(0,))
    grad_out_ptr = tl.make_block_ptr(grad_out_ptr, shape=(num_row,), 
                                strides=(grad_out_stride_row,),
                                offsets=(pid*ROW_BLOCK_SIZE,), 
                                block_shape=(ROW_BLOCK_SIZE,),
                                order=(0,))
    
    grad_x_ptr = tl.make_block_ptr(grad_x_ptr, shape=(num_row, d), 
                              strides=(grad_x_stride_row, grad_x_stride_dim),
                              offsets=(pid*ROW_BLOCK_SIZE, 0), 
                              block_shape=(ROW_BLOCK_SIZE, D_BLOCK_SIZE),
                              order=(1, 0))
    grad_w_partial_reduce_ptr = tl.make_block_ptr(grad_w_partial_reduce_ptr, shape=(num_row_block, d), 
                              strides=(grad_w_stride_row, grad_w_stride_dim),
                              offsets=(num_row_block, 0), 
                              block_shape=(1, D_BLOCK_SIZE),
                              order=(1, 0))

    for _ in range(0, d, D_BLOCK_SIZE):
        x_block = tl.load(x_ptr, boundary_check=(0, 1), padding_option='zero')
        w_block = tl.load(w_ptr, boundary_check=(0,), padding_option='zero')
        grad_out_block = tl.load(grad_out_block, boundary_check=(0,), padding_option='zero')
        
        tl.store(grad_x_ptr, grad_out_block[:, None] * w_block[None, :], boundary_check=(0, 1))
        tl.store(grad_w_partial_reduce_ptr, tl.sum(grad_out_block[:, None] * x_block, 0, keep_dims=True), boundary_check=(0,))

        x_block = tl.advance(x_block, (0, D_BLOCK_SIZE))
        w_block = tl.advance(w_block, (D_BLOCK_SIZE,))
        grad_x_ptr = tl.advance(grad_x_ptr, (0, D_BLOCK_SIZE))
        grad_w_partial_reduce_ptr = tl.advance(grad_w_partial_reduce_ptr, (0, D_BLOCK_SIZE))


class WeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor):
        ctx.shape_in = x.shape
        x = x.view(-1, ctx.shape_in[-1])
        num_row, d = x.shape
        ctx.ROW_BLOCK_SIZE = 16
        ctx.D_BLOCK_SIZE = triton.next_power_of_2(d) // 16
        ctx.save_for_backward(x, w)
        assert w.shape == x.shape[-1:]
        assert x.is_cuda and w.is_cuda
        assert x.is_contiguous()
        output = torch.empty(num_row, device=x.device)
        weighted_sum_forward[(triton.cdiv(num_row, ctx.ROW_BLOCK_SIZE),)](x, w, output, d, 1, 1, 1, num_row, d, ctx.ROW_BLOCK_SIZE, ctx.D_BLOCK_SIZE)
        return output.view(x.shape[:-1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor): # ctx: torch.autograd.function.FunctionCtx
        shape_in = ctx.shape_in
        x, w = ctx.saved_tensors
        x = x.view(-1, shape_in[-1])
        num_row, d = x.shape

        grad_out.view(-1)

        num_row_block = triton.cdiv(num_row, ctx.ROW_BLOCK_SIZE)
        grad_x = torch.empty(shape_in, device=grad_out.device)
        grad_w_partial_reduce = torch.empty((num_row_block, d), device=grad_out.device)
        weighted_sum_backward[(num_row_block,)](x, w, grad_out, grad_x, grad_w_partial_reduce, 
                                             d, 1, 1, 1, d, 1, d, 1, 
                                             num_row, d, ctx.ROW_BLOCK_SIZE, ctx.D_BLOCK_SIZE)
        grad_w = grad_w_partial_reduce.sum(0)
        return grad_x.view(*shape_in), grad_w
    
if __name__ == '__main__':
    device = 'cuda'
    batch = 1028
    n_elem = 4096
    x = torch.rand(batch, n_elem, device=device)
    w = torch.rand(n_elem, device=device)
    print(WeightedSum.apply(x, w))
    print((x * w[None, :]).sum(-1))