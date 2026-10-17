"""Render comparison or OURS-only GIFs from real four-view checkpoints.

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
def render_checkpoint(path, opt, dataset, angles, review_dir=None):
    model = FourViewLGM(opt).cuda().eval()
    mismatch = model.load_state_dict(load_file(path), strict=False)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(f'Checkpoint mismatch: {mismatch}')
    frames = {}
    metrics = []
    for index, subject in enumerate(dataset.items):
        data = batch(dataset, index)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gaussians = model.forward_gaussians(data['input'])
        if review_dir is not None:
            cameras = {key: data[key] for key in ('cam_view', 'cam_view_proj', 'cam_pos')}
            prediction = model.gs.render(gaussians, **cameras)['image'].float()
            target = data['images_output'].float()
            error = (prediction - target).square()
            mask = data['masks_output'].float()
            unseen = [1, 2, 4, 5, 7, 8, 10, 11]
            metrics.append({'subject': subject,
                'psnr_all': float(-10 * error.mean().clamp_min(1e-12).log10()),
                'psnr_unseen': float(-10 * error[:, unseen].mean().clamp_min(1e-12).log10()),
                'psnr_foreground': float(-10 * ((error * mask).sum() / (3 * mask.sum())).clamp_min(1e-12).log10())})
            sheet = Image.new('RGB', (opt.output_size * 12, opt.output_size * 2), 'white')
            for row, pixels in enumerate((target, prediction)):
                for view in range(12):
                    array = (pixels[0, view].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                    sheet.paste(Image.fromarray(array), (view * opt.output_size, row * opt.output_size))
            sheet.save(review_dir / f'review_{subject}.jpg', quality=95)
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
    if review_dir is not None:
        (review_dir / 'review_metrics.json').write_text(json.dumps(metrics, indent=2))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-checkpoint')
    parser.add_argument('--layout', choices=('comparison', 'ours'), default='comparison')
    parser.add_argument('--no-labels', action='store_true', help='Omit text headers and their reserved space')
    parser.add_argument('--review', action='store_true', help='Save 12-view reference comparisons and selection metrics locally')
    parser.add_argument('--ours-checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--subjects', nargs='+', default=['0515', '0516', '0518', '0513', '0521', '0512', '0520', '0519'])
    parser.add_argument('--dataset-name', default='THuman2.0', help='Source dataset recorded in the asset provenance')
    parser.add_argument('--output', default='assets/results')
    parser.add_argument('--ours-description', default='four-view cycle-adapted model')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--step-degrees', type=int, default=5)
    args = parser.parse_args()
    if args.layout == 'comparison' and not args.baseline_checkpoint:
        parser.error('comparison layout requires --baseline-checkpoint')
    if args.step_degrees <= 0 or 360 % args.step_degrees or args.resolution < 64:
        parser.error('step-degrees must divide 360; resolution must be >= 64')
    if any(not (500 <= int(s) <= 525) for s in args.subjects):
        parser.error('Choose loader IDs 0500..0525 from the supplied rendered-data directory')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    opt = copy.deepcopy(config_defaults['big'])
    opt.output_size = args.resolution
    opt.lambda_lpips = 0
    dataset = THuman(opt, split='test', training=False)
    dataset.root = str(Path(args.data_root).resolve())
    dataset.items = [f'{int(s):04d}' for s in args.subjects]
    angles = list(range(0, 360, args.step_degrees))
    baseline = render_checkpoint(args.baseline_checkpoint, opt, dataset, angles) if args.layout == 'comparison' else None
    ours = render_checkpoint(args.ours_checkpoint, opt, dataset, angles, output if args.review else None)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    size = args.resolution
    top = 12 if args.no_labels else 44
    width, height = (size * 2 + 36 if baseline is not None else size + 24), size + top + 12
    for subject in dataset.items:
        frames = []
        for index, right in enumerate(ours[subject]):
            canvas = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(canvas)
            if baseline is not None:
                canvas.paste(baseline[subject][index], (12, top))
                canvas.paste(right, (size + 24, top))
                if not args.no_labels:
                    draw.text((size / 2 + 12, 22), 'Baseline', font=font, anchor='mm', fill='#374151')
                    draw.text((size * 1.5 + 24, 22), 'OURS', font=font, anchor='mm', fill='#176752')
                draw.line((size + 18, top, size + 18, size + top), fill='#e5e7eb')
            else:
                canvas.paste(right, (12, top))
                if not args.no_labels:
                    draw.text((width / 2, 22), 'OURS', font=font, anchor='mm', fill='#176752')
            frames.append(canvas)
        # One palette per GIF reduces flicker and treats both methods equally.
        sample = Image.new('RGB', (width, height * 12), 'white')
        for index, frame in enumerate(frames[::max(1, len(frames) // 12)][:12]):
            sample.paste(frame, (0, index * height))
        palette = sample.quantize(colors=256)
        quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        prefix = 'comparison' if baseline is not None else 'ours'
        target = output / f'{prefix}_{subject}.gif'
        quantized[0].save(target, save_all=True, append_images=quantized[1:], duration=100,
                          loop=0, disposal=2, optimize=False)
        print(f'Saved {target} ({target.stat().st_size} bytes)', flush=True)
    metadata = {'dataset': args.dataset_name, 'subjects': dataset.items, 'frames': len(angles), 'step_degrees': args.step_degrees,
                'frame_duration_ms': 100, 'render_resolution': size,
                'source_views': [0, 3, 6, 9], 'crop': 'none', 'layout': args.layout, 'labels': not args.no_labels,
                'ours': args.ours_description,
                'ours_sha256': digest(args.ours_checkpoint)}
    if baseline is not None:
        metadata.update(baseline='local LGM initialization before adaptation', baseline_sha256=digest(args.baseline_checkpoint))
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
