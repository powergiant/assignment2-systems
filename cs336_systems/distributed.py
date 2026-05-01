import torch
import os

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import timeit

from typing import Iterable, Type

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
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
    def __init__(self, model: Module, rank: int, world_size: int):
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size
        # TODO: device
        self._broad_cast_parameters()
        self._adding_hooks()

    @torch.no_grad()
    def _broad_cast_parameters(self):
        for param in self.model.parameters():
            dist.broadcast(param, src=0)

    def _adding_hooks(self):
        def hook(param: Tensor):
            dist.all_reduce(param.grad)
            param.grad.div_(self.world_size)

        for param in self.model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

class SimpleNN(Module):
    def __init__(self, dim_in: int):
        super().__init__()
        self.dim_h = dim_in
        self.linear = nn.Linear(dim_in, 1, bias=False)
    
    def forward(self, x: Tensor):
        return self.linear.forward(x)

def test_ddp_naive(rank: int, world_size: int):
    setup(rank, world_size)
    simple_nn = SimpleNN(10)
    simple_nn_ddp = DDPNaive(simple_nn, rank, world_size)
    opt = torch.optim.Adam(simple_nn_ddp.parameters())

    for _ in range(5):
        simple_nn_ddp.zero_grad()
        data = torch.rand(10)
        loss: Tensor = simple_nn_ddp(data)
        loss.backward()
        opt.step()
        with torch.no_grad():
            dist.all_reduce(loss)
            loss.div_(world_size)
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
        offset += offset+tensor.numel()
    return outputs


class DDPFlat(Module):
    def __init__(self, model: Module, rank: int, world_size: int):
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size
        # TODO: device
        self._backward_buffer = []
        self._broad_cast_parameters()
        self._adding_hooks()

    @torch.no_grad()
    def _broad_cast_parameters(self):
        model_params = list(self.model.parameters())
        flatten_params = flatten_dense_tensors(model_params)
        dist.broadcast(flatten_params, src=0)
        unflatten_params = unflatten_dense_tensors(flatten_params, model_params)
        for param_b, param in zip(unflatten_params, model_params):
            param.copy_(param_b)
            
    def _adding_hooks(self):
        def hook():
            grads = []
            for param in self.model.parameters():
                if param.requires_grad:
                    grads.append(param.grad)
            flatten_grads = flatten_dense_tensors(grads)
            dist.all_reduce(flatten_grads)
            flatten_grads.div_(self.world_size)
            unflatten_grads = unflatten_dense_tensors(flatten_grads, grads)
            for grad_r, param in zip(unflatten_grads, self.model.parameters()):
                param.grad.copy_(grad_r)

        self.model.register_full_backward_hook(hook)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

def test_ddp_flat():
    pass

def run_test_ddp_flat():
    pass

class DDP(Module):
    def __init__(self, model: Module, rank: int, world_size: int):
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self._broadcast_paramters()

        self._handles = []

    @torch.no_grad()
    def _broadcast_paramters(self):
        model_params = list(self.model.parameters())
        flatten_params = flatten_dense_tensors(model_params)
        dist.broadcast(flatten_params, src=0)
        unflatten_params = unflatten_dense_tensors(flatten_params, model_params)
        for param_b, param in zip(unflatten_params, model_params):
            param.copy_(param_b)

    def _add_hooks(self):
        def hook(param: Tensor):
            self._handles.append((param, dist.all_reduce(param, async_op=True)))
        
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for param, handle in self._handles:
            handle.wait()
            param: Tensor
            param.grad.div_(self.world_size)
        self._handles.clear()

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
            # rank = id // 
            # param = 
            for param in self.param_groups[0]['param']:
                dist.broadcast(param, src=self.rank)
        pass

    def add_param_group(self, param_group):
        param_group_new = param_group.copy()
        param_group_new['param'] = list(param_group['param'])
        num_param_on_rank = len(param_group_new['param']) // self.world_size + 1
        param_group_new['param'] = param_group_new['param'][..., 
                        self.rank * num_param_on_rank : (self.rank + 1) * num_param_on_rank]
        self.optimizer_cls.add_param_group(self, param_group)
        
        

if __name__ == '__main__':
    test_simple_demo()
    # run_benchmark_all_reduce()
    run_test_ddp_naive()