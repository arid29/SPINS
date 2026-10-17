"""Render synchronized baseline/ours GIFs from two real four-view checkpoints.

No frame interpolation, generated imagery, or per-method cropping is applied.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from safetensors.torch import load_file
from core.options import config_defaults
from core.THuman_supervised_4_12 import THuman
from core.cycle_camera_schedule import shifted_cameras
from main_thuman_cycle_4views import FourViewLGM, batch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


@torch.no_grad()
def render_checkpoint(path, opt, dataset, angles):
    model = FourViewLGM(opt).cuda().eval()
    mismatch = model.load_state_dict(load_file(path), strict=False)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(f'Checkpoint mismatch: {mismatch}')
    frames = {}
    for index, subject in enumerate(dataset.items):
        data = batch(dataset, index)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gaussians = model.forward_gaussians(data['input'])
        images = []
        for angle in angles:
            cameras = shifted_cameras(data, angle, indices=(0,))
            pixels = model.gs.render(gaussians, **cameras)['image'][0, 0]
            array = (pixels.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            images.append(Image.fromarray(array))
        frames[subject] = images
        print(f'Rendered {subject}: {len(images)} views', flush=True)
    del model
    torch.cuda.empty_cache()
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--ours-checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--subjects', nargs='+', default=['0515', '0516', '0518', '0513', '0521', '0512', '0520', '0519'])
    parser.add_argument('--dataset-name', default='THuman2.0', help='Source dataset recorded in the asset provenance')
    parser.add_argument('--output', default='assets/results')
    parser.add_argument('--ours-description', default='four-view cycle-adapted model')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--step-degrees', type=int, default=5)
    args = parser.parse_args()
    if args.step_degrees <= 0 or 360 % args.step_degrees or args.resolution < 64:
        parser.error('step-degrees must divide 360; resolution must be >= 64')
    if any(not (500 <= int(s) <= 525) for s in args.subjects):
        parser.error('Choose held-out subjects 0500..0525')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    opt = copy.deepcopy(config_defaults['big'])
    opt.output_size = args.resolution
    opt.lambda_lpips = 0
    dataset = THuman(opt, split='test', training=False)
    dataset.root = str(Path(args.data_root).resolve())
    dataset.items = [f'{int(s):04d}' for s in args.subjects]
    angles = list(range(0, 360, args.step_degrees))
    baseline = render_checkpoint(args.baseline_checkpoint, opt, dataset, angles)
    ours = render_checkpoint(args.ours_checkpoint, opt, dataset, angles)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    size = args.resolution
    width, height = size * 2 + 36, size + 56
    for subject in dataset.items:
        frames = []
        for left, right in zip(baseline[subject], ours[subject]):
            canvas = Image.new('RGB', (width, height), 'white')
            canvas.paste(left, (12, 44))
            canvas.paste(right, (size + 24, 44))
            draw = ImageDraw.Draw(canvas)
            draw.text((size / 2 + 12, 22), 'Baseline', font=font, anchor='mm', fill='#374151')
            draw.text((size * 1.5 + 24, 22), 'OURS', font=font, anchor='mm', fill='#176752')
            draw.line((size + 18, 44, size + 18, size + 44), fill='#e5e7eb')
            frames.append(canvas)
        # One palette per GIF reduces flicker and treats both methods equally.
        sample = Image.new('RGB', (width, height * 12), 'white')
        for index, frame in enumerate(frames[::max(1, len(frames) // 12)][:12]):
            sample.paste(frame, (0, index * height))
        palette = sample.quantize(colors=256)
        quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        target = output / f'comparison_{subject}.gif'
        quantized[0].save(target, save_all=True, append_images=quantized[1:], duration=100,
                          loop=0, disposal=2, optimize=False)
        print(f'Saved {target} ({target.stat().st_size} bytes)', flush=True)
    metadata = {'dataset': args.dataset_name, 'subjects': dataset.items, 'frames': len(angles), 'step_degrees': args.step_degrees,
                'frame_duration_ms': 100, 'render_resolution': size,
                'source_views': [0, 3, 6, 9], 'crop': 'none; identical full frames for both methods',
                'baseline': 'local LGM initialization before adaptation',
                'ours': args.ours_description,
                'baseline_sha256': digest(args.baseline_checkpoint),
                'ours_sha256': digest(args.ours_checkpoint)}
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
