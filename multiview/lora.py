"""Small LoRA wrappers for upstream's custom multiview attention projections."""
import math

import torch
from torch import nn
from safetensors.torch import load_file, save_file


class LoRALinear(nn.Module):
    def __init__(self, base, rank=32, alpha=32):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        residual = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + residual * self.scale


def inject_lora(unet, rank=32):
    unet.requires_grad_(False)
    names = []
    for name, layer in list(unet.named_modules()):
        if isinstance(layer, nn.Linear) and ('attn1.' in name or 'attn2.' in name):
            parent, leaf = name.rsplit('.', 1)
            setattr(unet.get_submodule(parent), leaf, LoRALinear(layer, rank, rank))
            names.append(name)
    if not names:
        raise RuntimeError('No attention projections found')
    return names


def adapter_state(unet):
    return {name: p.detach().float().cpu().contiguous() for name, p in unet.named_parameters() if p.requires_grad}


def save_lora(unet, path):
    save_file(adapter_state(unet), str(path))


def load_lora(unet, path):
    state = load_file(str(path))
    expected = set(adapter_state(unet))
    if set(state) != expected:
        raise ValueError('LoRA checkpoint does not match the configured attention layers')
    unet.load_state_dict(state, strict=False)
