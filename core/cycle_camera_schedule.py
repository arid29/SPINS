"""Orbit-preserving camera offsets in the existing LGM renderer convention."""
import math
import torch


def maximum_offset(step, steps_per_epoch=500, epochs_per_stage=3):
    if step < 0 or steps_per_epoch < 1 or epochs_per_stage < 1:
        raise ValueError('Invalid schedule arguments')
    return min(90, 5 * (1 + step // (steps_per_epoch * epochs_per_stage)))


def shifted_cameras(data, degrees, indices=(0, 3, 6, 9)):
    """Rotate poses around the measured orbit, preserving orientation and radius.

    Infer the signed orbit axis from adjacent stored poses, so normalization and
    OpenGL/COLMAP conventions cannot silently reverse the requested offset.
    Rebuild renderer view, projection, and negative camera-center together.
    """
    with torch.autocast(device_type=data['cam_view'].device.type, enabled=False):
        view = data['cam_view'].float()
        poses = torch.linalg.inv(view.transpose(-1, -2))
        delta = poses[:, 1, :3, :3] @ poses[:, 0, :3, :3].transpose(-1, -2)
        axis = torch.stack((delta[:, 2, 1] - delta[:, 1, 2],
                            delta[:, 0, 2] - delta[:, 2, 0],
                            delta[:, 1, 0] - delta[:, 0, 1]), -1)
        norm = axis.norm(dim=-1, keepdim=True)
        if torch.any(norm < 1e-5):
            raise ValueError('Orbit axis is undefined')
        axis = axis / norm
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
        angle = math.radians(float(degrees))
        rotation = (torch.eye(3, device=view.device)[None]
                    + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew))
        center = poses[:, :, :3, 3].mean(1)
        selected = poses[:, indices].clone()
        selected[:, :, :3, :3] = rotation[:, None] @ selected[:, :, :3, :3]
        selected[:, :, :3, 3] = center[:, None] + (
            rotation[:, None] @ (selected[:, :, :3, 3] - center[:, None]).unsqueeze(-1)
        ).squeeze(-1)
        new_view = torch.linalg.inv(selected).transpose(-1, -2)
        projection = torch.linalg.solve(view[:, 0], data['cam_view_proj'][:, 0].float())
        return {'cam_view': new_view, 'cam_view_proj': new_view @ projection[:, None],
                'cam_pos': -selected[:, :, :3, 3]}
