import torch
import os

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Linear, Embedding
from torch.optim import Optimizer

import timeit

from typing import Iterable, Type

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)


def distributed_demo(rank: int, world_size: int):
    setup(rank, world_size)
    tensor_to_reduce = torch.randint(0, 4, (10,))
    print(f"The tensor on rank {rank} before reduce: {tensor_to_reduce}")
    dist.all_reduce(tensor_to_reduce, async_op=False)
    print(f"The tensor on rank {rank} after reduce: {tensor_to_reduce}")

def test_simple_demo():
    world_size = 4
    mp.spawn(fn = distributed_demo, args=(world_size,), nprocs= world_size, join=True)

def benchmark_all_reduce(rank: int, world_size: int):
    setup(rank, world_size)
    times = []
    for i in range(20):
        tensor_to_reduce = torch.randint(0, 4, (10,)).to(device=f'cuda:{rank}')
        if i < 5:
            dist.all_reduce(tensor_to_reduce, async_op=False)
            torch.cuda.synchronize()
        else:
            time_start = timeit.default_timer()
            dist.all_reduce(tensor_to_reduce, async_op=False)
            torch.cuda.synchronize()
            time_end = timeit.default_timer()
            times.append([i, time_end - time_start])
    if rank == 0:
        sum = 0
        for i, time in times:
            sum += time
        print(f"Average time of all-reduce: {sum/len(times)}")
        
def run_benchmark_all_reduce():
    world_size = 2
    mp.spawn(fn=benchmark_all_reduce, args=(world_size,), nprocs=world_size, join=True)

class DDPNaive(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        self._broadcast_parameters()
        self._adding_hooks()

    @torch.no_grad()
    def _broadcast_parameters(self):
        for param in self.module.parameters():
            dist.broadcast(param, src=0)

    def _adding_hooks(self):
        def hook(param: Tensor):
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        pass

class SimpleNN(Module):
    def __init__(self, dim_in: int):
        super().__init__()
        self.dim_h = dim_in
        self.linear = nn.Linear(dim_in, 1, bias=False)
    
    def forward(self, x: Tensor):
        return self.linear(x)

def test_ddp_naive(rank: int, world_size: int):
    setup(rank, world_size)
    is_cuda = True
    device = f'cuda:{rank}' if is_cuda else 'cpu'
    simple_nn = SimpleNN(10).to(device=device)
    simple_nn_ddp = DDPNaive(simple_nn)
    opt = torch.optim.Adam(simple_nn_ddp.parameters())

    for _ in range(5):
        simple_nn_ddp.zero_grad()
        data = torch.rand(10).to(device=device)
        loss: Tensor = simple_nn_ddp(data)
        loss.backward()
        opt.step()
        with torch.no_grad():
            dist.all_reduce(loss, dist.ReduceOp.AVG)
        if rank == 0:
            print(loss)

def run_test_ddp_naive():
    world_size = 2
    mp.spawn(fn=test_ddp_naive, args=(world_size,), nprocs=world_size, join=True)

def flatten_dense_tensors(tensors: Iterable[Tensor]) -> Tensor:
    return torch.cat([tensor.view(-1) for tensor in tensors])

def unflatten_dense_tensors(flatten_tensor: Tensor, tensors: Iterable[Tensor]) -> list[Tensor]:
    outputs = []
    offset = 0
    for tensor in tensors:
        outputs.append(flatten_tensor[offset: offset+tensor.numel()].view_as(tensor))
        offset += tensor.numel()
    return outputs


class DDPFlat(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        self._backward_buffer = []
        self._broadcast_parameters()

    @torch.no_grad()
    def _broadcast_parameters(self):
        module_params = list(self.module.parameters())
        flatten_params = flatten_dense_tensors(module_params)
        dist.broadcast(flatten_params, src=0)
        unflatten_params = unflatten_dense_tensors(flatten_params, module_params)
        for param_b, param in zip(unflatten_params, module_params):
            param.copy_(param_b)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        grads = []
        module_params = []
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad)
                module_params.append(param)
        if len(grads) == 0:
            return
        flatten_grads = flatten_dense_tensors(grads)
        dist.all_reduce(flatten_grads, op=dist.ReduceOp.AVG)
        unflatten_grads = unflatten_dense_tensors(flatten_grads, grads)
        for grad_r, param in zip(unflatten_grads, module_params):
            param.grad.copy_(grad_r)

def test_ddp_flat(rank: int, world_size: int):
    setup(rank, world_size)
    is_cuda = False
    device = f'cuda:{rank}' if is_cuda else 'cpu'
    simple_nn = SimpleNN(10).to(device=device)
    simple_nn_ddp = DDPFlat(simple_nn)
    opt = torch.optim.Adam(simple_nn_ddp.parameters())

    for _ in range(5):
        simple_nn_ddp.zero_grad()
        data = torch.rand(10).to(device=device)
        loss: Tensor = simple_nn_ddp(data)
        loss.backward()
        opt.step()
        with torch.no_grad():
            dist.all_reduce(loss, dist.ReduceOp.AVG)
        if rank == 0:
            print(loss)

def run_test_ddp_flat():
    world_size = 2
    mp.spawn(fn=test_ddp_flat, args=(world_size,), nprocs=world_size, join=True)

def test_flatten_unflatten():
    tensors = [torch.randint(0, 3, (5, 3)), torch.randint(0, 3, (2, 7))]
    assert torch.equal(tensors[1], unflatten_dense_tensors(flatten_dense_tensors(tensors), tensors)[1])

class DDP(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        self._broadcast_paramters()
        self._adding_hooks()

        self._handles = []

    @torch.no_grad()
    def _broadcast_parameters(self):
        module_params = list(self.module.parameters())
        flatten_params = flatten_dense_tensors(module_params)
        dist.broadcast(flatten_params, src=0)
        unflatten_params = unflatten_dense_tensors(flatten_params, module_params)
        for param_b, param in zip(unflatten_params, module_params):
            param.copy_(param_b)

    def _adding_hooks(self):
        def hook(param: Tensor):
            self._handles.append(dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True))
        
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self._handles:
            handle.wait()
        self._handles = []

def test_ddp():
    # model = SimpleNN().to(device)
    # ddp_model = DDP(`model)
    # for _ in range(train_steps):
    #     x, y = get_batch()
    #     logits = ddp_model(x)
    #     loss = loss_fn(logits, y)
    #     loss.backward()
    #     ddp_model.finish_gradient_synchronization()
    #     optimizer.step()`
    pass

def run_test_ddp():
    pass


class ShardedOptimizer(Optimizer):
    def __init__(self, params: Iterable[Tensor], optimizer_cls: Type[Optimizer], 
                 rank: int, world_size: int, **kwars):
        self.optimizer_cls = optimizer_cls
        self.rank = rank
        self.world_size = world_size
        self.params = list(params)
        self.sharding_axis = -1
        optimizer_cls.__init__(self, params, kwars)

    # def parameters(self):
    #     params = []
    #     for group in self.optimizer.param_groups:
    #         params.extend(group["params"])
    #     return params
    
    def step(self, closure = None):
        loss = self.optimizer_cls.step(self, closure)
        assert len(self.param_groups) == 1
        for id, param in enumerate(self.params):
            if id % self.world_size == self.rank:
                dist.broadcast(param, src=self.rank)
            else:
                continue
        return loss

    def add_param_group(self, param_group):
        param_group_new = param_group.copy()
        param_group_new['param'] = []
        for id in range(len(param_group_new['param'])):
            if id % self.world_size == self.rank:
                param_group_new['param'].append(param_group[id])
        self.optimizer_cls.add_param_group(self, param_group_new)
        
class FSDP(Module):
    def __init__(self, model: Module, wrap_modules: list[Linear | Embedding], rank: int, world_size: int, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.model = model
        self.wrap_modules = wrap_modules
        self.rank = rank
        self.world_size = world_size
        self._forward_handles = []
        self._backward_handles = []

    @torch.no_grad()
    def _split_param(self):
        for id, module in enumerate(self.wrap_modules):
            split_size = module.weight.shape[0] // self.world_size + 1
            module.weight.copy_(module.weight.split(split_size, dim=0)[self.rank])
            module.id = id

    def _add_forward_hook(self):
        if len(self.wrap_modules) <= 2:
            raise NotImplementedError("Not implemented for the case with <= 2 linears or embeddings")
        
        def pre_forward_hook(module: Linear | Embedding, input: Tensor):
            if module.id == 0:
                gather_ids = [0, 1, 2] 
            elif 0 < module.id < len(self.wrap_modules) - 2:
                gather_ids = [module.id + 2]
            else:
                gather_ids = []

            with torch.no_grad():
                for id in gather_ids:
                    module_gather = self.wrap_modules[id]
                    weight_shards = []
                    self._forward_handles.append(dist.all_gather(weight_shards, module_gather.weight, async_op=True))
                    module_gather.weight.copy_(torch.cat(weight_shards, dim=0))
            
            self._forward_handles[module.id].wait()

        def post_forward_hook(module: Linear | Embedding, input: Tensor, output: Tensor):
            with torch.no_grad():
                split_size = module.weight.shape[0] // self.world_size + 1
                module.weight.copy_(module.weight.split(split_size, dim=0)[self.rank])

        for module in self.wrap_modules:
            module.register_forward_pre_hook(pre_forward_hook)
            module.register_forward_hook(post_forward_hook)

    def _add_backward_hook(self):
        if len(self.wrap_modules) <= 2:
            raise NotImplementedError("Not implemented for the case with <= 2 linears or embeddings")
        
        def pre_backward_hook(module: Linear | Embedding, grad_output: Tensor):
            top_id = len(self.wrap_modules) - 1
            if module.id == top_id:
                gather_ids = [top_id, top_id - 1, top_id - 2] 
            elif 2 <= module.id < top_id:
                gather_ids = [module.id - 2]
            else:
                gather_ids = []

            with torch.no_grad():
                for id in gather_ids:
                    module_gather = self.wrap_modules[id]
                    weight_shards = []
                    self._forward_handles.append(dist.all_gather(weight_shards, module_gather.weight, async_op=True))
                    module_gather.weight.copy_(torch.cat(weight_shards, dim=0))
            
            self._backward_handles[module.id].wait()

        def post_backward_hook(module: Linear | Embedding, grad_input: Tensor, grad_output: Tensor):
            with torch.no_grad():
                split_size = module.weight.shape[0] // self.world_size + 1
                dist.reduce_scatter(module.weight, module.weight.split(split_size, dim=0))

        for module in self.wrap_modules:
            module.register_full_backward_pre_hook(pre_backward_hook)
            module.register_full_backward_hook(post_backward_hook)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self._forward_handles:
            handle.wait()
        self._forward_handles.clear()
        for handle in self._backward_handles:
            handle.wait()
        self._backward_handles.clear()
        for module in self.wrap_modules:
            module.weight.grad.div_(self.world_size)

if __name__ == '__main__':
    # test_simple_demo()
    # run_benchmark_all_reduce()
    # run_test_ddp_naive()
    # run_test_ddp_flat()
    test_flatten_unflatten()