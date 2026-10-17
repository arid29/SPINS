"""Camera geometry, renderer parity, GT isolation, and gradient smoke checks."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from core.cycle_camera_schedule import maximum_offset, shifted_cameras
from core.options import config_defaults
from core.THuman_supervised_4_12 import THuman
from main_thuman_cycle_4views import FourViewLGM, SOURCE, batch, cycle, features, render, save_grid
from cycle_progressive import cycle_degrees
from train_thuman_cycle_progressive import subject_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='THumanDataset/thuman2_rendered_lgm')
    parser.add_argument('--checkpoint', default='pretrained/model.safetensors')
    parser.add_argument('--workspace', default='workspace_smoke')
    args = parser.parse_args()
    output = Path(args.workspace)
    output.mkdir(exist_ok=True)
    torch.manual_seed(42)
    opt = copy.deepcopy(config_defaults['big'])
    opt.output_size = 256
    root = Path(args.data_root).resolve()
    raw = np.tile(np.eye(4), (526, 12, 1, 1))
    for subject in range(526):
        for view in range(12):
            raw[subject, view, :3] = np.load(root / 'parm' / f'{subject:04d}' / f'{view:02d}_extrinsic.npy')[:3]
    poses = torch.linalg.inv(torch.tensor(raw, dtype=torch.float32))
    poses[:, :, :3, 1:3] *= -1
    fixed = torch.eye(4)
    fixed[2, 3] = opt.cam_radius
    transform = fixed @ torch.linalg.inv(poses[:, 0])
    poses = transform[:, None] @ poses
    poses[:, :, :3, 1:3] *= -1
    ds = THuman(opt, split='train', training=False)
    ds.root = str(root)
    view = torch.linalg.inv(poses).transpose(-1, -2)
    data = {'cam_view': view, 'cam_view_proj': view @ ds.proj_matrix,
            'cam_pos': -poses[:, :, :3, 3]}
    report = {'geometry_subjects': 526, 'geometry_cameras': 6312, 'geometry_max_errors': {}}
    for degrees in [0, 30, 60, 90, -30, 360]:
        actual = shifted_cameras(data, degrees)
        ids = [(i + degrees // 30) % 12 for i in SOURCE]
        for key in actual:
            expected = data[key][:, ids]
            torch.testing.assert_close(actual[key], expected, atol=4e-6, rtol=4e-6)
            report['geometry_max_errors'][f'{degrees}_{key}'] = float((actual[key] - expected).abs().max())
    new = shifted_cameras(data, 5)
    c2w = torch.linalg.inv(new['cam_view'].transpose(-1, -2))
    rotation = c2w[:, :, :3, :3]
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation,
                               torch.eye(3).expand_as(rotation), atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(526, 4), atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(c2w[:, :, :3, 3].norm(dim=-1), torch.full((526, 4), opt.cam_radius), atol=3e-6, rtol=3e-6)
    for step, cap in [(0, 5), (1499, 5), (1500, 10), (25499, 85), (25500, 90), (29999, 90)]:
        assert maximum_offset(step) == cap
    train_ids, eval_ids = subject_split()
    assert train_ids == [f'{i:04d}' for i in range(500)]
    assert eval_ids == [f'{i:04d}' for i in range(500, 526)]
    print('PASS: all camera transforms, rotation/radius invariants, schedule boundaries, and split', flush=True)
    model = FourViewLGM(opt).cuda()
    incompatible = model.load_state_dict(load_file(args.checkpoint), strict=False)
    assert not incompatible.unexpected_keys
    assert all(k.startswith('lpips_loss.') for k in incompatible.missing_keys)
    sample = batch(ds, 0)
    for key in data:
        torch.testing.assert_close(sample[key].cpu(), data[key][0:1], atol=4e-6, rtol=4e-6)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        gs = model.forward_gaussians(sample['input'])
        for degrees in [0, 30, 90]:
            cameras = shifted_cameras(sample, degrees)
            ids = [(i + degrees // 30) % 12 for i in SOURCE]
            generated = model.gs.render(gs, **cameras)['image']
            stored = render(model, gs, sample, ids)['image']
            mae = float((generated - stored).abs().mean())
            assert mae < 5e-5, (degrees, mae)
            report[f'render_mae_{degrees}'] = mae
            a = features(model, generated, cameras['cam_view'])
            b = features(model, generated, sample['cam_view'][:, ids])
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
        old_loss, _, old_return, _ = cycle(model, sample, 1)
        loss, _, new_return, _ = cycle_degrees(model, sample, 30)
        assert abs(float(old_loss - loss)) < .002
        report['old_vs_new_cycle_loss_delta_30'] = float(abs(old_loss - loss))
        # GT at the other eight views must not influence training.
        changed = {k: v.clone() for k, v in sample.items()}
        unused = [i for i in range(12) if i not in SOURCE]
        changed['images_output'][:, unused] = float('nan')
        changed['masks_output'][:, unused] = float('nan')
        ref = cycle_degrees(model, sample, 5)[0]
        isolated = cycle_degrees(model, changed, 5)[0]
        torch.testing.assert_close(ref, isolated)
    print('PASS: renderer/ray parity and no intermediate GT in loss', flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=4e-6)
    report['gradient_steps'] = []
    for degrees in [5, -5, 10, 30, 60, 90]:
        model.train()
        model.lpips_loss.eval()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _, returned, synthetic = cycle_degrees(model, sample, degrees)
        assert torch.isfinite(loss)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        assert grad > 0
        optimizer.step()
        row = {'degrees': degrees, 'loss': float(loss.detach()), 'grad_norm': float(grad)}
        report['gradient_steps'].append(row)
        save_grid(output / f'cycle_{degrees:+03d}.jpg', [sample['images_output'][0, SOURCE], synthetic[0], returned['image'][0]])
        print(json.dumps(row), flush=True)
    report['status'] = 'passed'
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print('SMOKE PASSED', flush=True)


if __name__ == '__main__':
    main()
