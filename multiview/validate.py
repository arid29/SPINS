"""Camera and dataset checks before spending GPU time."""
import numpy as np

from .data import relative_cameras, target_views


def validate_dataset(dataset):
    max_position_error, max_pixel_error = 0., 0.
    expected_intrinsic = None
    for subject in dataset.subjects:
        poses, intrinsic = dataset.cameras[subject]
        if not np.allclose(np.linalg.det(poses[:, :3, :3]), 1., atol=1e-5):
            raise ValueError(f'Invalid camera rotations for {subject:04d}')
        if expected_intrinsic is None:
            expected_intrinsic = intrinsic[0]
        if not np.allclose(intrinsic, expected_intrinsic, atol=1e-3):
            raise ValueError('This configuration requires fixed intrinsics across the dataset')
        for view in range(12):
            for kind, suffix in [('img', 'jpg'), ('mask', 'png')]:
                path = dataset.root / kind / f'{subject:04d}' / f'{view:02d}.{suffix}'
                if not path.is_file():
                    raise FileNotFoundError(path)
            relative, transform = relative_cameras(poses, view)
            radius = np.linalg.norm(poses[view, :3, 3])
            positions = relative[target_views(view), :3, 3] / radius
            error = np.abs(positions - np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]])).max()
            max_position_error = max(max_position_error, float(error))
            if error > 1e-4:
                raise ValueError(f'Unexpected orbit direction or spacing: subject {subject}, source {view}')
            # Coordinate conversion must preserve projection of asymmetric 3D points.
            points = np.array([[.13, .21, -.08, 1], [-.23, -.11, .17, 1], [.1, -.3, .2, 1]]).T
            transformed = transform @ points
            for target in target_views(view):
                before = np.linalg.inv(poses[target]) @ points
                after = np.linalg.inv(relative[target]) @ transformed
                before[1:3] *= -1
                after[1:3] *= -1
                pixels_before = intrinsic[target] @ before[:3]
                pixels_after = intrinsic[target] @ after[:3]
                pixels_before /= pixels_before[2:3]
                pixels_after /= pixels_after[2:3]
                pixel_error = np.abs(pixels_before - pixels_after).max()
                max_pixel_error = max(max_pixel_error, float(pixel_error))
    if max_pixel_error > 1e-5:
        raise ValueError('Camera canonicalization changed pixel projections')
    return {'subjects': len(dataset.subjects), 'source_target_groups': len(dataset),
            'source_views_per_subject': 12, 'targets_per_source': 3,
            'source_02_targets': target_views(2), 'source_11_targets': target_views(11),
            'max_normalized_camera_position_error': max_position_error,
            'max_reprojection_error_pixels': max_pixel_error,
            'intrinsic_at_original_resolution': expected_intrinsic.tolist(),
            'occlusion_probability': dataset.occlusion_prob}
