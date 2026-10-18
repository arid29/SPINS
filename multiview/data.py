"""THuman images and calibrated, source-relative MV-Adapter conditions."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def target_views(source):
    if not 0 <= source < 12:
        raise ValueError('Source view must be in 0..11')
    return [(source + offset) % 12 for offset in (3, 6, 9)]


def load_cameras(root, subject):
    poses, intrinsics = [], []
    for view in range(12):
        folder = Path(root) / 'parm' / f'{subject:04d}'
        extrinsic = np.eye(4)
        extrinsic[:3] = np.load(folder / f'{view:02d}_extrinsic.npy')[:3]
        pose = np.linalg.inv(extrinsic)
        pose[:3, 1:3] *= -1  # OpenCV to OpenGL camera axes
        poses.append(pose)
        intrinsics.append(np.load(folder / f'{view:02d}_intrinsic.npy'))
    return np.stack(poses), np.stack(intrinsics)


def relative_cameras(poses, source):
    # MV-Adapter uses Z-up world axes, with the reference camera on negative Y.
    radius = np.linalg.norm(poses[source, :3, 3])
    canonical = np.array([[1., 0, 0, 0], [0, 0, -1, -radius],
                          [0, 1, 0, 0], [0, 0, 0, 1]])
    transform = canonical @ np.linalg.inv(poses[source])
    return transform @ poses, transform


def camera_condition(poses, size):
    # Preserve the pretrained adapter's six-channel direction/position convention.
    # Despite upstream calling it "plucker", this is NOT a per-pixel ray map.
    # Intrinsics are fixed throughout this experiment; perspective is learned from
    # the actual perspective target images, with no orthographic re-rendering.
    directions = -poses[:, :3, 2]
    positions = poses[:, :3, 3]
    positions = positions / np.linalg.norm(positions, axis=-1, keepdims=True)
    condition = np.concatenate((directions, positions), axis=-1)
    return torch.from_numpy(((condition + 1) / 2).astype(np.float32)).view(-1, 6, 1, 1).expand(-1, -1, size, size)


def occlude(image, foreground, rng, min_fraction=.10, max_fraction=.35):
    """Occlude a measured fraction of foreground, without changing framing."""
    ys, xs = np.where(foreground > .5)
    if len(xs) == 0:
        raise ValueError('Empty foreground mask')
    h, w = foreground.shape
    total = len(xs)
    bbox_w, bbox_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    for _ in range(100):
        center = int(rng.integers(total))
        rw = max(2, int(bbox_w * rng.uniform(.3, 1.1)))
        rh = max(2, int(bbox_h * rng.uniform(.12, .5)))
        x, y = int(xs[center]), int(ys[center])
        shape = Image.new('L', (w, h))
        draw = ImageDraw.Draw(shape)
        bounds = (x-rw//2, y-rh//2, x+rw//2, y+rh//2)
        (draw.rectangle if rng.random() < .6 else draw.ellipse)(bounds, fill=255)
        mask = np.asarray(shape) > 0
        fraction = float((mask & (foreground > .5)).sum() / total)
        if min_fraction <= fraction <= max_fraction:
            output = image.copy()
            if rng.random() < .5:
                fill = rng.uniform(0, 1, (1, 1, 3))
            else:
                patch = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
                fill = np.asarray(Image.fromarray(patch).resize((w, h), Image.Resampling.BILINEAR)) / 255
            output[mask] = np.broadcast_to(fill, image.shape)[mask]
            return output, mask, fraction
    # Deterministic fallback: a horizontal band covering a valid foreground count.
    counts = (foreground > .5).sum(axis=1)
    start = int(ys.min())
    end = start + int(np.searchsorted(np.cumsum(counts[start:]), total * .2)) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[start:end] = True
    fraction = float((mask & (foreground > .5)).sum() / total)
    if not min_fraction <= fraction <= max_fraction:
        raise ValueError('Could not construct bounded foreground occlusion')
    output = image.copy()
    output[mask] = rng.uniform(0, 1, 3)
    return output, mask, fraction


class THumanMultiview(Dataset):
    def __init__(self, root, split='train', size=768, occlusion_prob=.5, seed=42,
                 subjects=None):
        if split not in ('train', 'eval'):
            raise ValueError(split)
        self.root, self.size = Path(root), size
        allowed = range(500) if split == 'train' else range(500, 526)
        self.subjects = list(allowed if subjects is None else subjects)
        if not self.subjects or not set(self.subjects).issubset(allowed):
            raise ValueError('Subjects violate the train/evaluation split')
        if not 0 <= occlusion_prob <= 1:
            raise ValueError('Invalid occlusion probability')
        self.occlusion_prob, self.seed, self.epoch = occlusion_prob, seed, 0
        self.cameras = {s: load_cameras(root, s) for s in self.subjects}

    def __len__(self):
        return 12 * len(self.subjects)

    def read_view(self, subject, view):
        name = f'{subject:04d}/{view:02d}'
        with Image.open(self.root / 'img' / (name + '.jpg')) as im:
            original_size = im.size
            rgb = np.asarray(im.convert('RGB').resize((self.size, self.size), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        with Image.open(self.root / 'mask' / (name + '.png')) as im:
            mask = np.asarray(im.convert('L').resize((self.size, self.size), Image.Resampling.NEAREST)) > 0
        rgb = np.where(mask[..., None], rgb, 1.).astype(np.float32)
        return rgb, mask.astype(np.float32), original_size

    def __getitem__(self, index):
        subject, source = self.subjects[index // 12], index % 12
        targets = target_views(source)
        reference, mask, original_size = self.read_view(subject, source)
        clean = reference.copy()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        occ_mask, occ_fraction = np.zeros_like(mask), 0.
        if rng.random() < self.occlusion_prob:
            reference, occ_mask, occ_fraction = occlude(reference, mask, rng)
        target_images, target_masks = [], []
        for view in targets:
            rgb, target_mask, _ = self.read_view(subject, view)
            target_images.append(rgb)
            target_masks.append(target_mask)
        poses, intrinsic = self.cameras[subject]
        relative, transform = relative_cameras(poses, source)
        # Scaling the entire canvas preserves the principal point and field of view.
        intrinsic = intrinsic.copy()
        intrinsic[:, 0] *= self.size / original_size[0]
        intrinsic[:, 1] *= self.size / original_size[1]
        tensor_image = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32).copy()).permute(2, 0, 1)
        return {'reference': tensor_image(reference), 'reference_clean': tensor_image(clean),
                'targets': torch.stack([tensor_image(x) for x in target_images]),
                'target_masks': torch.from_numpy(np.stack(target_masks)),
                'control': camera_condition(relative[targets], self.size),
                'subject': subject, 'source': source, 'target_views': torch.tensor(targets),
                'occlusion_mask': torch.from_numpy(np.asarray(occ_mask, dtype=np.float32)),
                'occlusion_fraction': occ_fraction,
                'c2w': torch.from_numpy(relative[[source] + targets].astype(np.float32)),
                'intrinsic': torch.from_numpy(intrinsic[[source] + targets].astype(np.float32)),
                'world_transform': torch.from_numpy(transform.astype(np.float32))}
