"""
This small-subset trial is not a full training run.
"""

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file, save_file

from core.models_selfsup import LGM
from core.options import config_defaults
from core.THuman_supervised_4_12 import THuman
from core.unet import MVAttention
from core.utils import get_rays

SOURCE = [0, 3, 6, 9]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FourViewLGM(LGM):
    def __init__(self, opt):
        super().__init__(opt)
        for module in self.unet.modules():
            if isinstance(module, MVAttention):
                module.num_frames = 4

    def forward_gaussians(self, images):
        b, v, c, h, w = images.shape
        assert v == 4, f"Expected four distinct views, got {v}"
        x = self.conv(self.unet(images.reshape(b * v, c, h, w)))
        x = x.reshape(b, v, 14, self.opt.splat_size, self.opt.splat_size)
        x = x.permute(0, 1, 3, 4, 2).reshape(b, -1, 14)
        return torch.cat((self.pos_act(x[..., :3]), self.opacity_act(x[..., 3:4]),
                          self.scale_act(x[..., 4:7]), self.rot_act(x[..., 7:11]),
                          self.rgb_act(x[..., 11:])), dim=-1)


def features(model, rgb, cam_view):
    """Match dataset channels: normalized RGB first, then Pluecker rays."""
    b, v = rgb.shape[:2]
    rgb = F.interpolate(rgb.flatten(0, 1), (model.opt.input_size,) * 2,
                        mode='bilinear', align_corners=False)
    rgb = TF.normalize(rgb, MEAN, STD)
    c2w = torch.linalg.inv(cam_view.flatten(0, 1).transpose(-2, -1))
    c2w[:, :3, 1:3] *= -1  # COLMAP -> OpenGL
    rays = []
    for pose in c2w:
        origin, direction = get_rays(pose, model.opt.input_size,
                                     model.opt.input_size, model.opt.fovy)
        rays.append(torch.cat((torch.cross(origin, direction, dim=-1), direction), -1))
    rays = torch.stack(rays).permute(0, 3, 1, 2)
    return torch.cat((rgb, rays), 1).reshape(b, v, 9, model.opt.input_size, model.opt.input_size)


def render(model, gaussians, data, indices):
    return model.gs.render(gaussians, **{
        key: data[key][:, indices] for key in ('cam_view', 'cam_view_proj', 'cam_pos')
    })


def cycle(model, data, offset):
    novel = [(i + offset) % 12 for i in SOURCE]
    with torch.no_grad():
        first = model.forward_gaussians(data['input'])
        synthetic = render(model, first, data, novel)['image']
    second_input = features(model, synthetic.detach(), data['cam_view'][:, novel])
    second = model.forward_gaussians(second_input)
    returned = render(model, second, data, SOURCE)
    target = data['images_output'][:, SOURCE]
    mask = data['masks_output'][:, SOURCE]
    rgb_mse = F.mse_loss(returned['image'], target)
    alpha_mse = F.mse_loss(returned['alpha'], mask)
    lpips = rgb_mse.new_zeros(())
    if model.opt.lambda_lpips:
        pred = F.interpolate(returned['image'].flatten(0, 1), (256, 256), mode='bilinear', align_corners=False)
        gt = F.interpolate(target.flatten(0, 1), (256, 256), mode='bilinear', align_corners=False)
        lpips = model.lpips_loss(pred * 2 - 1, gt * 2 - 1).mean()
    loss = rgb_mse + alpha_mse + model.opt.lambda_lpips * lpips
    return loss, {'loss': loss, 'rgb_mse': rgb_mse, 'alpha_mse': alpha_mse, 'lpips': lpips}, returned, synthetic


def batch(dataset, index):
    return {key: value.unsqueeze(0).cuda() for key, value in dataset[index].items()}


def save_grid(path, rows):
    # Each row contains ordered views. Images are kept at render resolution.
    pixels = torch.cat([torch.cat(list(row.detach().float().cpu()), dim=-1) for row in rows], dim=-2)
    array = (pixels.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


@torch.no_grad()
def evaluate(model, datasets, output, tag):
    model.eval()
    destination = output / tag
    destination.mkdir()
    records = []
    for split, dataset, count in datasets:
        for index in range(count):
            data = batch(dataset, index)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                gs = model.forward_gaussians(data['input'])
                direct = render(model, gs, data, list(range(12)))
                _, components, returned, synthetic = cycle(model, data, 1)
            mse = (direct['image'].float() - data['images_output']).square().mean((0, 2, 3, 4))
            novel = [i for i in range(12) if i not in SOURCE]
            record = {'split': split, 'subject': dataset.items[index],
                      'cycle': {k: float(v) for k, v in components.items()},
                      'direct_psnr_all': float(-10 * mse.mean().log10()),
                      'direct_psnr_input': float(-10 * mse[SOURCE].mean().log10()),
                      'direct_psnr_unseen': float(-10 * mse[novel].mean().log10()),
                      'gaussians': gs.shape[1], 'mean_opacity': float(gs[..., 3].mean())}
            records.append(record)
            name = f'{split}_{dataset.items[index]}'
            save_grid(destination / f'{name}_direct.jpg', [data['images_output'][0], direct['image'][0]])
            save_grid(destination / f'{name}_cycle.jpg', [data['images_output'][0, SOURCE], synthetic[0], returned['image'][0]])
            print(json.dumps({'evaluation': tag, **record}), flush=True)
    (destination / 'metrics.json').write_text(json.dumps(records, indent=2))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--checkpoint', default='pretrained/model.safetensors')
    parser.add_argument('--data-root', default='THumanDataset/thuman2_rendered_lgm')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--train-subjects', type=int, default=4)
    parser.add_argument('--eval-subjects', type=int, default=2)
    parser.add_argument('--output-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=4e-6)
    parser.add_argument('--lambda-lpips', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    assert args.steps >= 0 and 1 <= args.train_subjects <= 500 and 1 <= args.eval_subjects <= 26
    output = Path(args.workspace).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists():
        raise FileExistsError(f'Refusing to overwrite a previous trial: {output}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    opt = copy.deepcopy(config_defaults['big'])
    opt.output_size = args.output_size
    opt.num_input_views = 4
    opt.num_views = 12
    opt.lambda_lpips = args.lambda_lpips
    opt.prob_grid_distortion = opt.prob_cam_jitter = 0
    model = FourViewLGM(opt).cuda()
    weights = load_file(args.checkpoint, device='cpu')
    incompatible = model.load_state_dict(weights, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith('lpips_loss.')]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f'Checkpoint mismatch: {missing}, {incompatible.unexpected_keys}')
    del weights
    train = THuman(opt, split='train', training=False)
    test = THuman(opt, split='test', training=False)
    train.root = test.root = str(Path(args.data_root).resolve())
    config = {**vars(args), 'options': asdict(opt), 'source_views': SOURCE,
              'cycle_camera_sets': [[1, 4, 7, 10], [2, 5, 8, 11]],
              'train_ids': train.items[:args.train_subjects], 'test_ids': test.items[:args.eval_subjects],
              'gpu': torch.cuda.get_device_name(), 'checkpoint_realpath': str(Path(args.checkpoint).resolve()),
              'image_rows': {'direct': ['GT 00..11', 'direct prediction 00..11'],
                             'cycle': ['GT source 00,03,06,09', 'synthetic 01,04,07,10', 'cycle return 00,03,06,09']}}
    source_files = [Path(__file__), Path('core/models_selfsup.py'), Path('core/unet.py'),
                    Path('core/THuman_supervised_4_12.py'), Path('core/options.py')]
    config['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    # Rebuilt features must reproduce the dataset's RGB/ray convention. At 256
    # input/output sizes these two resize paths coincide exactly up to roundoff.
    sample = batch(train, 0)
    rebuilt = features(model, sample['images_output'][:, SOURCE], sample['cam_view'][:, SOURCE])
    feature_error = float((rebuilt - sample['input']).abs().max())
    if args.output_size == opt.input_size:
        torch.testing.assert_close(rebuilt, sample['input'], atol=2e-5, rtol=2e-5)
    print(json.dumps({'feature_max_error': feature_error, 'config': config}), flush=True)
    del sample, rebuilt
    datasets = [('train', train, min(2, args.train_subjects)), ('test', test, args.eval_subjects)]
    before = evaluate(model, datasets, output, 'before')
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, weight_decay=.05, betas=(.9, .95))
    started = time.monotonic()
    with (output / 'train.jsonl').open('w', buffering=1) as log:
        for step in range(args.steps):
            model.train()
            if hasattr(model, 'lpips_loss'):
                model.lpips_loss.eval()
            index = step % args.train_subjects
            data = batch(train, index)
            offset = 1 + random.randrange(2)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, components, _, _ = cycle(model, data, offset)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if not grad > 0:
                raise RuntimeError('No training gradient')
            optimizer.step()
            record = {'step': step + 1, 'subject': train.items[index], 'offset': offset,
                      **{k: float(v.detach()) for k, v in components.items()}, 'grad_norm': float(grad),
                      'elapsed_seconds': time.monotonic() - started,
                      'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30}
            log.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
    after = evaluate(model, datasets, output, 'after')
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(output / 'model.safetensors'))
    summary = {'status': 'complete', 'steps': args.steps, 'before': before, 'after': after,
               'elapsed_seconds': time.monotonic() - started}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'COMPLETE: {output}', flush=True)


if __name__ == '__main__':
    main()
