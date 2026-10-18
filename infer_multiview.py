"""Generate three calibrated THuman target views from one conditioning image."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'pretrained' / 'huggingface'))

import numpy as np
import torch
from PIL import Image

from multiview.data import THumanMultiview, camera_condition
from multiview.lora import inject_lora, load_lora
from multiview.model import build_pipeline, prompt_embeddings, generate
from train_multiview_lora import save_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', help='Path to step directory; omit for pretrained baseline')
    parser.add_argument('--data-root')
    parser.add_argument('--subject', type=int, default=500)
    parser.add_argument('--source', type=int, default=2)
    parser.add_argument('--image', help='Preframed square RGB image; alpha is composited onto white')
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--occluded', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    config = json.loads((Path(args.adapter) / 'config.json').read_text()) if args.adapter else {}
    size = config.get('size', args.size)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.image:
        if args.occluded:
            parser.error('--occluded applies to dataset inputs with foreground masks')
        with Image.open(args.image) as im:
            if im.width != im.height:
                parser.error('Provide a square, preframed image; stretching changes camera calibration')
            rgb = np.asarray(im.convert('RGBA').resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        rgb = rgb[..., :3] * rgb[..., 3:4] + 1 - rgb[..., 3:4]
        reference = torch.from_numpy(rgb).permute(2, 0, 1)
        positions = np.array([[1., 0, 0], [0, 1, 0], [-1, 0, 0]])
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, :3, 0] = np.cross(np.array([0., 0, 1]), positions)
        poses[:, :3, 1] = np.array([0., 0, 1])
        poses[:, :3, 2] = positions
        poses[:, :3, 3] = positions * 1.5
        control = camera_condition(poses, size)
        sample = None
    else:
        if not args.data_root:
            parser.error('--data-root or --image is required')
        if not 0 <= args.source < 12:
            parser.error('--source must be 0..11')
        dataset = THumanMultiview(args.data_root, split='eval' if args.subject >= 500 else 'train',
                                 size=size, subjects=[args.subject], seed=args.seed,
                                 occlusion_prob=float(args.occluded))
        sample = dataset[args.source]
        reference, control = sample['reference'], sample['control']
    pipe = build_pipeline()
    embeddings = prompt_embeddings(pipe)
    if args.adapter:
        inject_lora(pipe.unet, config['rank'])
        load_lora(pipe.unet, Path(args.adapter) / 'adapter.safetensors')
    pipe.unet.eval()
    preds = generate(pipe, embeddings, reference, control, seed=args.seed, steps=args.steps)
    Image.fromarray((reference.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(out / 'input.png')
    for degree, image in zip((90, 180, 270), preds):
        image.save(out / f'view_{degree:03d}.png')
    metadata = {'source': args.source if sample else None, 'relative_target_degrees': [90, 180, 270],
                'target_views': sample['target_views'].tolist() if sample else None,
                'size': size, 'seed': args.seed, 'projection': 'THuman fixed perspective',
                'occlusion_fraction': sample['occlusion_fraction'] if sample else None}
    if sample:
        save_grid(sample, out / 'comparison.png', preds)
        np.savez(out / 'cameras.npz', c2w=sample['c2w'].numpy(), intrinsic=sample['intrinsic'].numpy(),
                 world_transform=sample['world_transform'].numpy())
        pred = np.stack([np.asarray(im, dtype=np.float32) / 255 for im in preds])
        gt = sample['targets'].permute(0, 2, 3, 1).numpy()
        mask = sample['target_masks'].numpy()[..., None]
        mse = ((pred - gt)**2).mean((1, 2, 3))
        fg_mse = (((pred - gt)**2) * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * 3)
        metadata['psnr_per_target'] = (-10 * np.log10(mse.clip(1e-10))).tolist()
        metadata['foreground_psnr_per_target'] = (-10 * np.log10(fg_mse.clip(1e-10))).tolist()
    (out / 'metrics.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
