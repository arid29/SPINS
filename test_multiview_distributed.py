"""Compare a two-worker LoRA update against the equivalent global batch."""
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from multiview.distributed import average_gradients
from multiview.lora import inject_lora


def make_model():
    torch.manual_seed(19)
    model = nn.Module()
    model.attn1 = nn.Sequential(nn.Linear(4, 3))
    model.attn2 = nn.Sequential(nn.Linear(4, 3))
    model.unused = nn.Module()
    model.unused.attn1 = nn.Sequential(nn.Linear(4, 3))
    inject_lora(model, rank=2)
    return model


def worker(rank, init_path, output):
    dist.init_process_group('gloo', init_method=f'file://{init_path}', rank=rank, world_size=2)
    model = make_model()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.1)
    # Each rank contributes two microbatches; attn2 is unused on rank zero.
    for micro in range(2):
        x = torch.arange(4).float().reshape(1, 4) + rank + micro
        output_tensor = model.attn1(x)
        if rank:
            output_tensor = output_tensor + model.attn2(x)
        (output_tensor.square().mean() / 2).backward()
    average_gradients(params, bucket_elements=7)
    optimizer.step()
    torch.save(model.state_dict(), Path(output) / f'rank{rank}.pt')
    dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_accumulated_update_matches_global_batch_and_replicas(self):
        model = make_model()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.1)
        for rank in range(2):
            for micro in range(2):
                x = torch.arange(4).float().reshape(1, 4) + rank + micro
                prediction = model.attn1(x)
                if rank:
                    prediction = prediction + model.attn2(x)
                (prediction.square().mean() / 4).backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker, args=(str(Path(folder) / 'rendezvous'), folder), nprocs=2, join=True)
            for rank in range(2):
                result = torch.load(Path(folder) / f'rank{rank}.pt', weights_only=True)
                for key, expected in model.state_dict().items():
                    torch.testing.assert_close(result[key], expected)


if __name__ == '__main__':
    unittest.main()
