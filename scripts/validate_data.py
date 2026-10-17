"""Validate rendered THuman inputs before starting GPU training."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=525)
    args = parser.parse_args()
    if not 0 <= args.start <= args.end <= 525:
        parser.error('Expected 0 <= start <= end <= 525')
    for number in range(args.start, args.end + 1):
        sid = f'{number:04d}'
        poses = []
        for view in range(12):
            with Image.open(args.data_root / 'img' / sid / f'{view:02d}.jpg') as im:
                size = im.size
                im.verify()
            with Image.open(args.data_root / 'mask' / sid / f'{view:02d}.png') as mask:
                if mask.size != size or not np.asarray(mask).any():
                    raise ValueError(f'Mask shape/foreground invalid: {sid}/{view:02d}')
            if size[0] != size[1]:
                raise ValueError('Expected square input images')
            matrix = np.load(args.data_root / 'parm' / sid / f'{view:02d}_extrinsic.npy')
            if matrix.shape not in ((3, 4), (4, 4)) or not np.isfinite(matrix).all():
                raise ValueError(f'Bad extrinsic: {sid}/{view:02d}')
            e = np.eye(4); e[:3] = matrix[:3]
            np.testing.assert_allclose(e[:3, :3] @ e[:3, :3].T, np.eye(3), atol=1e-5)
            np.testing.assert_allclose(np.linalg.det(e[:3, :3]), 1., atol=1e-5)
            poses.append(np.linalg.inv(e))
            path = args.data_root / 'parm' / sid / f'{view:02d}_intrinsic.npy'
            if path.exists():
                k = np.load(path)
                f = size[0] / 2 / np.tan(np.deg2rad(49.1) / 2)
                expected = np.array([[f, 0., size[0]/2], [0., f, size[1]/2], [0., 0., 1.]])
                np.testing.assert_allclose(k, expected, rtol=1e-3, atol=1e-5)
        poses = np.stack(poses)
        centers = poses[:, :3, 3]
        np.testing.assert_allclose(centers.mean(0), 0., atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.5, atol=1e-5)
        np.testing.assert_allclose(centers[:, 1], 0., atol=1e-5)
        np.testing.assert_allclose(poses[:, :3, 2], -centers / 1.5, atol=1e-5)
        for view in range(12):
            angle = np.deg2rad(view * 30)
            orbit = np.eye(4)
            orbit[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
            np.testing.assert_allclose(orbit @ poses[0], poses[view], atol=1e-5)
    print(f'PASS: subjects {args.start:04d}–{args.end:04d}; image/mask/camera layout and 30-degree orbit')


if __name__ == '__main__':
    main()
