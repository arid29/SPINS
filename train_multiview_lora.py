"""Train an image-conditioned MV-Adapter LoRA on THuman's 12-view orbits."""
import argparse
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'pretrained' / 'huggingface'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from multiview.data import THumanMultiview
from multiview.lora import inject_lora, load_lora, save_lora
from multiview.model import build_pipeline, prompt_embeddings, generate, UPSTREAM_REVISION


def save_grid(sample, path, predictions=None):
    pictures = [sample['reference_clean'], sample['reference']] + list(sample['targets'])
    images = [Image.fromarray((x.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)) for x in pictures]
    size = images[0].width
    canvas = Image.new('RGB', (size * 5, size * (2 if predictions else 1)), 'white')
    for i, im in enumerate(images):
        canvas.paste(im, (i * size, 0))
    if predictions:
        for i, im in enumerate(predictions):
            canvas.paste(im, ((i + 2) * size, size))
    canvas.save(path)


def save_training(pipe, optimizer, args, step, consumed, workspace):
    folder = workspace / f'step_{step:06d}'
    folder.mkdir(exist_ok=True)
    save_lora(pipe.unet, folder / 'adapter.safetensors')
    torch.save({'optimizer': optimizer.state_dict(), 'step': step, 'consumed': consumed,
                'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                'config': vars(args)}, folder / 'training.pt')
    (folder / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    (workspace / 'latest.txt').write_text(str(folder.resolve()) + '\n')
    return folder


def preview(pipe, embeds, datasets, args, workspace, step):
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    pipe.unet.eval()
    metrics = {'step': step, 'subject': 500, 'source': 2, 'target_views': [5, 8, 11]}
    try:
        for label, evaluation in datasets:
            sample = evaluation[2]
            preds = generate(pipe, embeds, sample['reference'], sample['control'],
                             steps=args.preview_steps, seed=args.seed)
            save_grid(sample, workspace / f'preview_{step:06d}_{label}.png', preds)
            prediction = np.stack([np.asarray(im, dtype=np.float32) / 255 for im in preds])
            truth = sample['targets'].permute(0, 2, 3, 1).numpy()
            mask = sample['target_masks'].numpy()[..., None]
            squared = (prediction - truth)**2
            mse = squared.mean((1, 2, 3))
            foreground_mse = (squared * mask).sum((1, 2, 3)) / (3 * mask.sum((1, 2, 3)))
            metrics[label] = {'psnr': (-10 * np.log10(mse.clip(1e-10))).tolist(),
                              'foreground_psnr': (-10 * np.log10(foreground_mse.clip(1e-10))).tolist(),
                              'occlusion_fraction': sample['occlusion_fraction']}
    finally:
        pipe.unet.train()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
    (workspace / f'preview_{step:06d}_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps({'preview': metrics}), flush=True)


def loss_step(pipe, embeds, sample, device):
    dtype = pipe.unet.dtype
    target_rgb = sample['targets'].to(device, dtype)
    reference = sample['reference'].unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(target_rgb * 2 - 1).latent_dist.sample() * pipe.vae.config.scaling_factor
        ref_latent = pipe.vae.encode(reference * 2 - 1).latent_dist.sample() * pipe.vae.config.scaling_factor
        dropped = bool(torch.rand((), device=device) < .1)
        prompt = embeds['negative_prompt_embeds' if dropped else 'prompt_embeds']
        pooled = embeds['negative_pooled_prompt_embeds' if dropped else 'pooled_prompt_embeds']
        size = reference.shape[-1]
        time_ids = torch.tensor([[size, size, 0, 0, size, size]], device=device, dtype=dtype)
        ref_states = {}
        # Reference features follow the frozen/detached reference branch upstream.
        pipe.unet(ref_latent, torch.zeros(1, device=device, dtype=torch.long),
                  encoder_hidden_states=prompt, added_cond_kwargs={'text_embeds': pooled, 'time_ids': time_ids},
                  cross_attention_kwargs={'cache_hidden_states': ref_states, 'use_mv': False, 'use_ref': False})
        ref_states = {k: (torch.zeros_like(v) if dropped else v).repeat_interleave(3, 0) for k, v in ref_states.items()}
        features = pipe.cond_encoder(sample['control'].to(device, dtype))
    noise = torch.randn_like(latent)
    scheduler = pipe.training_scheduler
    timestep = torch.randint(scheduler.config.num_train_timesteps, (1,), device=device).repeat(3)
    noisy = scheduler.add_noise(latent, noise, timestep)
    predicted = pipe.unet(noisy, timestep, encoder_hidden_states=prompt.repeat(3, 1, 1),
        added_cond_kwargs={'text_embeds': pooled.repeat(3, 1), 'time_ids': time_ids.repeat(3, 1)},
        down_intrablock_additional_residuals=list(features),
        cross_attention_kwargs={'ref_hidden_states': ref_states, 'num_views': 3}).sample
    prediction_type = scheduler.config.prediction_type
    if prediction_type == 'epsilon':
        target = noise
    elif prediction_type == 'v_prediction':
        target = scheduler.get_velocity(latent, noise, timestep)
    else:
        raise ValueError(prediction_type)
    alpha = scheduler.alphas_cumprod.to(device)[timestep].float()
    snr = alpha / (1 - alpha).clamp_min(1e-8)
    denominator = snr + 1 if prediction_type == 'v_prediction' else snr
    weights = snr.clamp(max=5) / denominator.clamp_min(1e-8)
    per_view = F.mse_loss(predicted.float(), target.float(), reduction='none').mean((1, 2, 3))
    return (per_view * weights).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--workspace', default='workspace_multiview_lora')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--occlusion-prob', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--preview-every', type=int, default=500)
    parser.add_argument('--preview-steps', type=int, default=30)
    parser.add_argument('--train-subjects', type=int, nargs='+', help='Optional training subset for short checks')
    parser.add_argument('--resume', help='Path to a step directory')
    parser.add_argument('--check-only', action='store_true', help='Validate cameras and preview data without a GPU')
    args = parser.parse_args()
    if args.size < 64 or args.size % 64 or min(args.steps, args.accumulation, args.rank, args.save_every, args.preview_steps) < 1 or args.preview_every < 0:
        parser.error('size must be a positive multiple of 64; step counts, rank, and accumulation must be positive')
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if (workspace / 'config.json').exists() and not args.resume and not args.check_only:
        raise FileExistsError('Use a new workspace or --resume')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = THumanMultiview(args.data_root, size=args.size, occlusion_prob=args.occlusion_prob,
                             seed=args.seed, subjects=args.train_subjects)
    from multiview.validate import validate_dataset
    report = validate_dataset(dataset)
    (workspace / 'data_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    for i in (0, 2, 11):
        save_grid(dataset[i], workspace / f'data_source_{i:02d}.png')
    print(json.dumps(report), flush=True)
    if args.check_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for training')
    device = 'cuda'
    pipe = build_pipeline()
    embeds = prompt_embeddings(pipe)
    names = inject_lora(pipe.unet, args.rank)
    pipe.unet.enable_gradient_checkpointing()
    pipe.unet.train()
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    config = {**vars(args), 'train_subjects_count': len(dataset.subjects), 'examples_per_epoch': len(dataset),
              'trainable_parameters': sum(p.numel() for p in params), 'lora_layers': names,
              'upstream_revision': UPSTREAM_REVISION, 'gpu': torch.cuda.get_device_name(),
              'targets': '(source + [3,6,9]) % 12', 'camera_condition': 'native direction-position, source relative',
              'projection': 'fixed calibrated THuman perspective', 'occlusion_foreground_fraction': [.1, .35]}
    start_step, consumed = 0, 0
    if args.resume:
        folder = Path(args.resume)
        state = torch.load(folder / 'training.pt', map_location='cpu', weights_only=False)
        for key in ('size', 'rank', 'lr', 'accumulation', 'occlusion_prob', 'seed', 'data_root', 'train_subjects'):
            if state['config'][key] != vars(args)[key]:
                raise ValueError(f'Resume config differs: {key}')
        load_lora(pipe.unet, folder / 'adapter.safetensors')
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        start_step, consumed = state['step'], state['consumed']
    (workspace / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    eval_clean = THumanMultiview(args.data_root, split='eval', size=args.size, occlusion_prob=0, subjects=[500])
    eval_occluded = THumanMultiview(args.data_root, split='eval', size=args.size, occlusion_prob=1, subjects=[500])
    previews = [('clean', eval_clean), ('occluded', eval_occluded)]
    if not args.resume:
        preview(pipe, embeds, previews, args, workspace, 0)
    order, epoch_cached = None, -1
    last_folder = None
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.steps + 1):
        started, loss_sum, occ_sum = time.monotonic(), 0., 0.
        warmup = min(1., step / min(100, max(1, args.steps // 10)))
        for group in optimizer.param_groups:
            group['lr'] = args.lr * warmup
        for micro in range(args.accumulation):
            epoch, offset = divmod(consumed, len(dataset))
            if epoch != epoch_cached:
                dataset.epoch = epoch
                order = np.random.default_rng(args.seed + epoch).permutation(len(dataset))
                epoch_cached = epoch
            sample = dataset[int(order[offset])]
            # The reference pass is detached but shares the LoRA modules with
            # the target pass. Do not reuse no-grad autocast weight casts when
            # checkpointing recomputes the differentiable target branch.
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                loss = loss_step(pipe, embeds, sample, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            (loss / args.accumulation).backward()
            loss_sum += loss.detach().item() / args.accumulation
            occ_sum += sample['occlusion_fraction'] / args.accumulation
            consumed += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(grad_norm) or grad_norm == 0:
            raise FloatingPointError(f'Invalid LoRA gradients: {grad_norm}')
        if any(p.grad is not None for p in pipe.unet.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen base received gradients')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        log = {'step': step, 'examples_seen': consumed, 'loss': loss_sum,
               'grad_norm': grad_norm.item(), 'occlusion_fraction': occ_sum,
               'seconds': time.monotonic() - started,
               'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2**30}
        with (workspace / 'train.jsonl').open('a') as f:
            f.write(json.dumps(log) + '\n')
        print(json.dumps(log), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            last_folder = save_training(pipe, optimizer, args, step, consumed, workspace)
        if (args.preview_every and step % args.preview_every == 0) or step == args.steps:
            preview(pipe, embeds, previews, args, workspace, step)
    # Strictly reload the saved adapter to verify the exported parameter set.
    if last_folder:
        load_lora(pipe.unet, last_folder / 'adapter.safetensors')
    print('TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
