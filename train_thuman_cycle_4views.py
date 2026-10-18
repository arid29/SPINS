

import argparse
import copy
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from main_thuman_cycle_4views import FourViewLGM, SOURCE, batch, cycle, evaluate, features
from core.options import config_defaults
from core.THuman_supervised_4_12 import THuman


def subject_split():
    train = [f'{i:04d}' for i in range(500)]
    evaluation = [f'{i:04d}' for i in range(500, 526)]
    assert len(train) == 500 and len(evaluation) == 26
    assert not set(train).intersection(evaluation)
    return train, evaluation


def preflight(root, subjects):
    missing = []
    for subject in subjects:
        for view in range(12):
            for folder, name in [('img', f'{view:02d}.jpg'), ('mask', f'{view:02d}.png'),
                                 ('parm', f'{view:02d}_extrinsic.npy')]:
                path = root / folder / subject / name
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing dataset files: {missing[:10]}')


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def save_weights(path, weights):
    temporary = path.with_name(path.name + '.tmp')
    save_file(weights, str(temporary))
    os.replace(temporary, path)


def aggregate(records):
    result = {}
    for split in ('train', 'test'):
        selected = [r for r in records if r['split'] == split]
        if not selected:
            continue
        result[split] = {
            'subjects': len(selected),
            'cycle_loss': float(np.mean([r['cycle']['loss'] for r in selected])),
            'direct_psnr_all': float(np.mean([r['direct_psnr_all'] for r in selected])),
            'direct_psnr_unseen': float(np.mean([r['direct_psnr_unseen'] for r in selected])),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--checkpoint', default='pretrained/model.safetensors')
    parser.add_argument('--data-root', default='THumanDataset/thuman2_rendered_lgm')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=4e-6)
    parser.add_argument('--lambda-lpips', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--selection-metric', choices=['direct_psnr_all', 'direct_psnr_unseen'], default='direct_psnr_all')
    parser.add_argument('--resume', action='store_true', help='Resume latest training_state.pt in this workspace')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    output = Path(args.workspace).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists() and not args.resume:
        raise FileExistsError(f'Use a new workspace or --resume: {output}')
    train_ids, eval_ids = subject_split()
    root = Path(args.data_root).resolve()
    preflight(root, train_ids + eval_ids)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    opt = copy.deepcopy(config_defaults['big'])
    opt.output_size = 256
    opt.num_input_views, opt.num_views = 4, 12
    opt.batch_size = 1
    opt.num_epochs, opt.lr = args.epochs, args.lr
    opt.workspace, opt.resume = str(output), str(Path(args.checkpoint).resolve())
    opt.lambda_lpips = args.lambda_lpips
    opt.prob_grid_distortion = opt.prob_cam_jitter = 0
    train = THuman(opt, split='train', training=False)
    evaluation = THuman(opt, split='test', training=False)
    # Override the historical loader's 0000..0504 / 0505..0525 split explicitly.
    train.items, evaluation.items = train_ids, eval_ids
    train.root = evaluation.root = str(root)
    source_files = [Path(__file__), Path('main_thuman_cycle_4views.py'),
                    Path('core/THuman_supervised_4_12.py'), Path('core/models_selfsup.py'),
                    Path('core/unet.py'), Path('core/options.py')]
    config = {**vars(args), 'options': asdict(opt), 'train_ids': train_ids, 'eval_ids': eval_ids,
              'data_root': str(root), 'source_views': SOURCE, 'steps_per_epoch': len(train_ids),
              'total_steps': args.epochs * len(train_ids), 'gpu': torch.cuda.get_device_name(),
              'cycle_camera_sets': [[1, 4, 7, 10], [2, 5, 8, 11]],
              'best_metric': args.selection_metric + ' (mean evaluation PSNR; higher is better)',
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    if args.resume:
        original = json.loads((output / 'config.json').read_text())
        for key in ('train_ids', 'eval_ids', 'data_root', 'lr', 'lambda_lpips', 'selection_metric', 'seed', 'source_sha256'):
            if original[key] != config[key]:
                raise ValueError(f'Resume configuration differs: {key}')
        atomic_json(output / 'resume_config.json', config)
    else:
        atomic_json(output / 'config.json', config)
        snapshot = output / 'source'
        snapshot.mkdir()
        for path in source_files:
            (snapshot / path.name).write_bytes(path.read_bytes())
    print(json.dumps({'split': {'train': [train_ids[0], train_ids[-1], len(train_ids)],
                               'eval': [eval_ids[0], eval_ids[-1], len(eval_ids)]},
                      'epochs': args.epochs, 'total_steps': config['total_steps']}), flush=True)
    model = FourViewLGM(opt).cuda()
    state = torch.load(output / 'training_state.pt', map_location='cpu', weights_only=False) if args.resume else None
    weights = state['model'] if state else load_file(args.checkpoint, device='cpu')
    incompatible = model.load_state_dict(weights, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith('lpips_loss.')]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f'Checkpoint mismatch: {missing}, {incompatible.unexpected_keys}')
    del weights
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, weight_decay=.05, betas=(.9, .95))
    start_epoch, global_step, best_psnr = 0, 0, -float('inf')
    if state:
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, global_step, best_psnr = state['epoch'], state['step'], state['best_psnr']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    sample = batch(train, 0)
    rebuilt = features(model, sample['images_output'][:, SOURCE], sample['cam_view'][:, SOURCE])
    torch.testing.assert_close(rebuilt, sample['input'], atol=2e-5, rtol=2e-5)
    del sample, rebuilt
    datasets = [('train', train, 2), ('test', evaluation, len(eval_ids))]
    if not args.resume:
        baseline = aggregate(evaluate(model, datasets, output, 'before'))
        atomic_json(output / 'baseline.json', baseline)
        best_psnr = baseline['test'][args.selection_metric]
        save_weights(output / 'best.safetensors', {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()})
        atomic_json(output / 'best.json', {'epoch': 0, 'step': 0, 'metrics': baseline})
    started = time.monotonic()
    with (output / 'train.jsonl').open('a', buffering=1) as log, (output / 'epochs.jsonl').open('a', buffering=1) as epoch_log:
        for epoch in range(start_epoch, args.epochs):
            order = list(range(len(train)))
            random.shuffle(order)
            epoch_loss = 0.
            for index in order:
                model.train()
                if hasattr(model, 'lpips_loss'):
                    model.lpips_loss.eval()
                data = batch(train, index)
                offset = 1 + random.randrange(2)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, components, _, _ = cycle(model, data, offset)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {global_step}')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if not grad > 0:
                    raise RuntimeError('No training gradient')
                optimizer.step()
                global_step += 1
                epoch_loss += float(loss.detach())
                record = {'epoch': epoch + 1, 'step': global_step, 'subject': train.items[index], 'offset': offset,
                          **{k: float(v.detach()) for k, v in components.items()}, 'grad_norm': float(grad),
                          'elapsed_seconds': time.monotonic() - started,
                          'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30}
                log.write(json.dumps(record) + '\n')
                if global_step == 1 or global_step % 25 == 0:
                    print(json.dumps(record), flush=True)
                    atomic_json(output / 'progress.json', {'status': 'running', **record,
                                                           'total_steps': config['total_steps']})
            metrics = aggregate(evaluate(model, datasets, output, f'epoch_{epoch + 1:03d}'))
            epoch_record = {'epoch': epoch + 1, 'step': global_step,
                            'mean_training_loss': epoch_loss / len(train), 'metrics': metrics}
            epoch_log.write(json.dumps(epoch_record) + '\n')
            print(json.dumps({'epoch_complete': epoch_record}), flush=True)
            weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
            score = metrics['test'][args.selection_metric]
            if score > best_psnr:
                best_psnr = score
                save_weights(output / 'best.safetensors', weights)
                atomic_json(output / 'best.json', epoch_record)
            save_weights(output / 'model.safetensors', weights)
            if (epoch + 1) % 5 == 0:
                save_weights(output / f'model_epoch_{epoch + 1:03d}.safetensors', weights)
            state = {'model': weights, 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                     'step': global_step, 'best_psnr': best_psnr, 'python_rng': random.getstate(),
                     'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all()}
            temporary = output / 'training_state.pt.tmp'
            torch.save(state, temporary)
            os.replace(temporary, output / 'training_state.pt')
            del state, weights
            atomic_json(output / 'progress.json', {'status': 'running', 'completed_epoch': epoch + 1,
                                                   'step': global_step, 'total_steps': config['total_steps'],
                                                   'latest_metrics': metrics})
    atomic_json(output / 'summary.json', {'status': 'complete', 'epochs': args.epochs,
                                        'steps': global_step, 'best': json.loads((output / 'best.json').read_text())})
    atomic_json(output / 'progress.json', {'status': 'complete', 'completed_epoch': args.epochs, 'step': global_step})
    print(f'COMPLETE: {output}', flush=True)


if __name__ == '__main__':
    main()
