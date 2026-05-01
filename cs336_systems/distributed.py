import torch
import os

import torch.distributed as dist
import torch.multiprocessing as mp

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

if __name__ == '__main__':
    test_simple_demo()