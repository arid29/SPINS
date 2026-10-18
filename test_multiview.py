import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from multiview.data import target_views, occlude, relative_cameras
from multiview.lora import inject_lora, save_lora, load_lora


class MultiviewTests(unittest.TestCase):
    def test_shared_detached_branch_with_checkpointed_autocast(self):
        model = nn.Module()
        model.attn1 = nn.Sequential(nn.Linear(6, 8), nn.SiLU(), nn.Linear(8, 6))
        inject_lora(model, rank=2)
        x = torch.randn(4, 6)
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            with torch.no_grad():
                reference = model.attn1(x)
            target = checkpoint(model.attn1, x, use_reentrant=False)
            loss = (target.float() - reference.float() + .1).square().mean()
        loss.backward()
        for name, p in model.named_parameters():
            if name.endswith('lora_B'):
                self.assertIsNotNone(p.grad)
                self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_all_source_target_groups(self):
        for source in range(12):
            self.assertEqual([(t-source) % 12 for t in target_views(source)], [3, 6, 9])
            self.assertNotIn(source, target_views(source))
        self.assertEqual(target_views(2), [5, 8, 11])
        self.assertEqual(target_views(11), [2, 5, 8])

    def test_occlusion_is_bounded_and_does_not_mutate_clean_input(self):
        image = np.ones((128, 128, 3), dtype=np.float32)
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[12:116, 45:83] = 1
        for seed in range(50):
            modified, occ, fraction = occlude(image, mask, np.random.default_rng(seed))
            self.assertGreaterEqual(fraction, .1)
            self.assertLessEqual(fraction, .35)
            np.testing.assert_array_equal(modified[~occ], image[~occ])
            np.testing.assert_array_equal(image, 1.)
            self.assertGreater(np.abs(modified - image).sum(), 0)

    def test_lora_zero_initialization_gradients_and_round_trip(self):
        model = nn.Module()
        model.attn1 = nn.Sequential(nn.Linear(6, 8), nn.SiLU(), nn.Linear(8, 6))
        x = torch.randn(4, 6)
        expected = model.attn1(x).detach()
        original = {k: p.clone() for k, p in model.named_parameters()}
        inject_lora(model, rank=2)
        torch.testing.assert_close(model.attn1(x), expected)
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        model.attn1(x).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        optim.step()
        for name, value in original.items():
            parent, leaf = name.rsplit('.', 1)
            torch.testing.assert_close(model.get_parameter(parent + '.base.' + leaf), value)
        updated = model.attn1(x).detach()
        self.assertFalse(torch.equal(updated, expected))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'adapter.safetensors'
            save_lora(model, path)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.zero_()
            load_lora(model, path)
            torch.testing.assert_close(model.attn1(x), updated)


if __name__ == '__main__':
    unittest.main()
