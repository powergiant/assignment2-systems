import torch
import os

import torch.distributed as dist
import torch.multiprocessing as mp

import timeit

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
            times.append([i, time_start - time_end])
    if rank == 0:
        sum = 0
        for i, time in times:
            sum += time
        print(f"Average time of all-reduce: {sum/len(times)}")
        
def run_benchmark_all_reduce():
    world_size = 2
    mp.spawn(fn=benchmark_all_reduce, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    test_simple_demo()