import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter
# from options import Options
# from utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class THuman(Dataset):

    def __init__(self, opt: Options, split= 'train', training=True):

        self.opt = opt
        self.split = split
        self.training = training

        self.root = 'THumanDataset/thuman2_rendered_lgm'

        # Build IDs: '0000' ... '0525'
        ids = [f"{i:04d}" for i in range(526)]  # 0..525 inclusive

        # Explicit research split: train 0000..0499, eval 0500..0525
        train_ids = ids[:500]   # first 500 → train
        test_ids  = ids[500:]   # remaining 26 → evaluation

        if self.split == 'train':
            self.items = train_ids
        else:
            self.items = test_ids


        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        return len(self.items)

    def __call__(self, idx):
        return self.__getitem__(idx)

    def getWorld2View2(self, R, t, translate=np.array([.0, .0, .0]), scale=1.0):

        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center

        return np.float32(C2W)


    def __getitem__(self, idx):

        subject = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []


        for i in range(self.opt.num_views):

            image_path = os.path.join(self.root, 'img',  subject, f'{i:02d}.jpg')
            mask_path = os.path.join(self.root, 'mask', subject,  f'{i:02d}.png')
            camera_path = os.path.join(self.root, 'parm', subject,  f'{i:02d}_extrinsic.npy')

            # read RGB image (float32 in [0,1])
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0  # [H,W,3]

            # read mask (binary float32 in {0,1}); default to ones if missing
            try:
                mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
                mask = (mask > 0).astype(np.float32)  # [H,W]
            except Exception:
                mask = np.ones(image.shape[:2], dtype=np.float32)

            # black bg -> white bg using mask (foreground=1 keeps image; background=0 becomes white)
            image = image * mask[..., None] + (1.0 - mask[..., None])

            # load extrinsics and convert to c2w
            cam_Es = np.load(camera_path)  # expected (3,4) or (4,4)

            R = np.array(cam_Es[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            t = np.array(cam_Es[:3, 3], np.float32)
            c2w = self.getWorld2View2(R, t)


            # OpenCV --> OpenGL
            c2w[:3, 1:3] *= -1

            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale

            # convert to torch tensors with expected shapes
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous()   # [3,H,W]
            mask = torch.from_numpy(mask).unsqueeze(0).contiguous()         # [1,H,W]
            c2w = torch.from_numpy(c2w).to(torch.float32).contiguous()      # [4,4]


            images.append(image)
            masks.append(mask)
            cam_poses.append(c2w)


        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, C, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]


        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]



        if self.training:

            ## add random rotation to overall video
            V = images.shape[0]
            start = torch.randint(0, V, ()).item()   # e.g., 5 -> order becomes 5,6,7,...,V-1,0,1,...,4
            perm = torch.cat([torch.arange(start, V), torch.arange(0, start)])

            images    = images.index_select(0, perm)      # [V, C, H, W]
            masks     = masks.index_select(0, perm)       # [V, C, H, W]
            cam_poses = cam_poses.index_select(0, perm)   # [V, 4, 4]
            ###

        input_indices = torch.tensor([0,3,6,9])

        images_input = F.interpolate(images[input_indices].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[input_indices].clone()


        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)


        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results


if __name__ == "__main__":

    opt = Options()
    ds = THuman(opt, split='train', training=True)
    abc = ds(0)
