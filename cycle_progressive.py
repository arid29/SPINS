"""Cycle-only loss using analytically shifted cameras; no intermediate GT."""
import torch
import torch.nn.functional as F
from core.cycle_camera_schedule import shifted_cameras
from main_thuman_cycle_4views import SOURCE, features, render


def cycle_degrees(model, data, degrees):
    cameras = shifted_cameras(data, degrees)
    with torch.no_grad():
        first = model.forward_gaussians(data['input'])
        synthetic = model.gs.render(first, **cameras)['image']
    second_input = features(model, synthetic.detach(), cameras['cam_view'])
    second = model.forward_gaussians(second_input)
    returned = render(model, second, data, SOURCE)
    target = data['images_output'][:, SOURCE]
    mask = data['masks_output'][:, SOURCE]
    rgb_mse = F.mse_loss(returned['image'], target)
    alpha_mse = F.mse_loss(returned['alpha'], mask)
    lpips = rgb_mse.new_zeros(())
    if model.opt.lambda_lpips:
        pred = F.interpolate(returned['image'].flatten(0, 1), (256, 256), mode='bilinear', align_corners=False)
        gt = F.interpolate(target.flatten(0, 1), (256, 256), mode='bilinear', align_corners=False)
        lpips = model.lpips_loss(pred * 2 - 1, gt * 2 - 1).mean()
    loss = rgb_mse + alpha_mse + model.opt.lambda_lpips * lpips
    return loss, {'loss': loss, 'rgb_mse': rgb_mse, 'alpha_mse': alpha_mse, 'lpips': lpips}, returned, synthetic
