"""Synchronous LoRA gradient averaging for torchrun, without wrapping the UNet.

The reference and target calls share a UNet, but only the target call has a
backward pass. Synchronizing at the optimizer boundary also supports gradient
accumulation without communicating the frozen model.
"""
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def initialize():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    return rank, world_size, f'cuda:{local_rank}'


def barrier():
    if dist.is_initialized():
        if dist.get_backend() == 'nccl':
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def average_gradients(parameters, bucket_elements=4_000_000):
    """Average accumulated gradients, including parameters unused on a rank."""
    if not dist.is_initialized():
        return
    parameters = list(parameters)
    present = torch.tensor([p.grad is not None for p in parameters],
                           device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(present, op=dist.ReduceOp.MAX)
    bucket, elements = [], 0

    def reduce_bucket():
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat)
        flat.div_(dist.get_world_size())
        offset = 0
        for parameter in bucket:
            count = parameter.numel()
            parameter.grad.copy_(flat[offset:offset + count].view_as(parameter))
            offset += count

    for parameter, used in zip(parameters, present.tolist()):
        if not used:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        bucket.append(parameter)
        elements += parameter.numel()
        if elements >= bucket_elements:
            reduce_bucket()
            bucket, elements = [], 0
    if bucket:
        reduce_bucket()


def mean_values(values, device):
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor)
        tensor.div_(dist.get_world_size())
    return tensor.tolist()


def gather_rng():
    state = {'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}
    if not dist.is_initialized():
        return [state]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, state)
    return states
