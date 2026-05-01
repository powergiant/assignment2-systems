import triton
import triton.language as tl
import torch
import math

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
                              offsets=(pid*ROW_BLOCK_SIZE, 0), 
                              block_shape=(ROW_BLOCK_SIZE, D_BLOCK_SIZE),
                              order=(1, 0))
    w_ptr = tl.make_block_ptr(w_ptr, shape=(d, ), 
                              strides=(w_stride_dim,),
                              offsets=(0,), 
                              block_shape=(D_BLOCK_SIZE,),
                              order=(0,))
    out_ptr = tl.make_block_ptr(out_ptr, shape=(num_row,), 
                                strides=(out_stride_row,),
                                offsets=(pid*ROW_BLOCK_SIZE,), 
                                block_shape=(ROW_BLOCK_SIZE,),
                                order=(0,))
    
    output = tl.zeros((ROW_BLOCK_SIZE,), dtype=tl.float32)

    for _ in range(0, d, D_BLOCK_SIZE):
        x_block = tl.load(x_ptr, boundary_check=(0, 1), padding_option='zero')
        w_block = tl.load(w_ptr, boundary_check=(0,), padding_option='zero')
        output += tl.sum(x_block * w_block[None, :], 1)

        x_ptr = x_ptr.advance((0, D_BLOCK_SIZE))
        w_ptr = w_ptr.advance((D_BLOCK_SIZE,))

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

        x_ptr = tl.advance(x_ptr, (0, D_BLOCK_SIZE))
        w_ptr = tl.advance(w_ptr, (D_BLOCK_SIZE,))
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
    
def test_weighted_sum_triton():
    device = 'cuda'
    batch = 1028
    n_elem = 4096
    x = torch.rand(batch, n_elem, device=device)
    w = torch.rand(n_elem, device=device)
    print(WeightedSum.apply(x, w))
    print((x * w[None, :]).sum(-1))

@triton.jit
def flash_attention_forward(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
                            stride_qb, stride_qq, stride_qd,
                            stride_kb, stride_kk, stride_kd,
                            stride_vb, stride_vk, stride_vd,
                            stride_ob, stride_oq, stride_od,
                            stride_lb, stride_ld,
                            T_queries, T_keys, 
                            D: tl.constexpr,
                            Q_BLOCK_SIZE: tl.constexpr, K_BLOCK_SIZE: tl.constexpr,
                            scale):
    query_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_ptr = tl.make_block_ptr(Q_ptr + batch_idx * stride_qb, 
                              shape = (T_queries, D),
                              strides = (stride_qq, stride_qd),
                              offsets = (query_idx * Q_BLOCK_SIZE, 0),
                              block_shape = (Q_BLOCK_SIZE, D),
                              order = (1, 0))
    K_ptr = tl.make_block_ptr(K_ptr + batch_idx * stride_kb, 
                              shape = (T_keys, D),
                              strides = (stride_kk, stride_kd),
                              offsets = (0, 0),
                              block_shape = (K_BLOCK_SIZE, D),
                              order = (1, 0))
    V_ptr = tl.make_block_ptr(V_ptr + batch_idx * stride_vb, 
                              shape = (T_keys, D),
                              strides = (stride_vk, stride_vd),
                              offsets = (0, 0),
                              block_shape = (K_BLOCK_SIZE, D),
                              order = (1, 0))
    O_ptr = tl.make_block_ptr(O_ptr + batch_idx * stride_ob, 
                              shape = (T_queries, D),
                              strides = (stride_oq, stride_od),
                              offsets = (query_idx * Q_BLOCK_SIZE, D),
                              block_shape = (Q_BLOCK_SIZE, D),
                              order = (1, 0))
    L_ptr = tl.make_block_ptr(L_ptr + batch_idx * stride_lb, 
                              shape = (T_queries,),
                              strides = (stride_ld),
                              offsets = (query_idx * Q_BLOCK_SIZE,),
                              block_shape = (Q_BLOCK_SIZE,),
                              order = (0,))
    
    
    Q_block = tl.load(Q_ptr, boundary_check=(0, 1), padding_option='zero')
    O_block = tl.zeros((Q_BLOCK_SIZE, D), tl.float32)
    L_block = tl.zeros((Q_BLOCK_SIZE,), tl.float32)
    M_block = tl.full((Q_BLOCK_SIZE,), float('-inf'), tl.float32)

    for _ in range(tl.cdiv(T_keys, K_BLOCK_SIZE)):
        K_block = tl.load(K_ptr, boundary_check=(0, 1), padding_option='zero')
        V_block = tl.load(V_ptr, boundary_check=(0, 1), padding_option='zero')
        S_block = tl.dot(Q_block, K_block.T) * scale
        row_max = tl.max(S_block, axis=1)
        M_block_prev = M_block
        M_block = tl.maximum(M_block_prev, row_max)
        P_block = tl.exp(S_block - M_block[:, None])
        L_block = tl.exp(M_block_prev - M_block) * L_block + tl.sum(P_block, 1)
        O_block = tl.exp(M_block_prev - M_block)[:, None] *  O_block + tl.dot(P_block, V_block)

        tl.store(O_ptr, O_block, boundary_check=(0, 1))
        tl.store(L_ptr, L_block, boundary_check=(0,))

        K_ptr = tl.advance(K_ptr, (K_BLOCK_SIZE, 0))
        V_ptr = tl.advance(V_ptr, (K_BLOCK_SIZE, 0))

    O_block = O_block / (L_block[:, None])
    L_block = M_block + tl.log(L_block)
    tl.store(O_ptr, O_block, boundary_check=(0, 1))
    tl.store(L_ptr, L_block, boundary_check=(0,))

    
@triton.jit
def flash_attention_backward():
    raise NotImplementedError("The backward pass of flash attention is not implemented")

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        # TODO: shape check, cuda check, contiguous check
        B, T_queries, D = Q.shape
        T_keys = K.shape[1]

        Q_BLOCK_SIZE = 64
        K_BLOCK_SIZE = triton.next_power_of_2(T_keys) // 16
        
        num_b_block = tl.cdiv(B, 4)
        num_q_block = tl.cdiv(T_queries, Q_BLOCK_SIZE)
        O = torch.empty_like(Q)
        L = torch.empty((B, D), dtype=Q.dtype, device=Q.device)
        flash_attention_forward[(num_b_block, num_q_block)](Q, K, V, O, L, 
                    T_queries * D, D, 1,
                    T_keys * D, D, 1,
                    T_keys * D, D, 1,
                    T_queries * D, D, 1,
                    D, 1, 
                    T_queries, T_keys, D, 
                    Q_BLOCK_SIZE, K_BLOCK_SIZE,
                    scale=1/math.sqrt(D))
        return O
        
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        return super().backward(ctx, *grad_outputs)

if __name__ == '__main__':
    from torch.nn. functional import softmax
    import math
    device = 'cuda'
    B, T_queries, T_keys, D = 128, 1024, 2048, 512
    Q = torch.rand(B, T_queries, B)
    K = torch.rand(B, T_keys, B)
    V = torch.rand(B, T_keys, B)
    
    print(FlashAttention.apply(Q, K, V))
    print(softmax(Q @ K.transpose(-1, -2) / math.sqrt(D)) @ V)