"""Approximate THuman textured-scan rendering for SPINS (not paper preprocessing).

Uses a 12-camera horizontal orbit, height normalization, perspective projection,
and depth-derived masks. EGL is the default headless OpenGL backend.
"""
import argparse
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def camera_pose(degrees, radius=1.5):
    angle = math.radians(degrees)
    position = np.array([radius * math.sin(angle), 0., radius * math.cos(angle)])
    backward = position / np.linalg.norm(position)
    right = np.cross([0., 1., 0.], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack((right, up, backward), axis=1)
    pose[:3, 3] = position
    return pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-root', type=Path, required=True, help='Contains 0000/0000.obj, MTL, textures, etc.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=525, help='Inclusive last subject ID')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--smplx-root', type=Path, help='Optional trusted local fittings: ID/smplx_param.pkl')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.start <= args.end <= 525 or args.resolution < 64:
        parser.error('Expected 0 <= start <= end <= 525 and resolution >= 64')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import pyglet
    pyglet.options['shadow_window'] = False
    import pyrender
    import trimesh

    radius, fovy, views = 1.5, 49.1, 12
    size = args.resolution
    focal = .5 * size / math.tan(math.radians(fovy) / 2)
    intrinsic = np.array([[focal, 0., size / 2], [0., focal, size / 2], [0., 0., 1.]])
    renderer = pyrender.OffscreenRenderer(size, size)
    try:
        for number in range(args.start, args.end + 1):
            subject = f'{number:04d}'
            folders = {key: args.output / key / subject for key in ('img', 'mask', 'parm')}
            expected = [folders['img'] / f'{i:02d}.jpg' for i in range(views)]
            expected += [folders['mask'] / f'{i:02d}.png' for i in range(views)]
            expected += [folders['parm'] / f'{i:02d}_{kind}.npy' for i in range(views) for kind in ('extrinsic', 'intrinsic')]
            metadata = folders['parm'] / 'preprocess.json'
            if any(p.exists() for p in expected):
                if args.skip_existing and all(p.exists() for p in expected) and metadata.exists():
                    print(f'Skip complete subject {subject}', flush=True)
                    continue
                raise FileExistsError(f'Existing or partial subject {subject}; use a fresh output directory')
            source = args.mesh_root / subject / f'{subject}.obj'
            scene_mesh = trimesh.load(str(source), force='scene', process=False)
            meshes = []
            for node in scene_mesh.graph.nodes_geometry:
                transform, geometry = scene_mesh.graph[node]
                mesh = scene_mesh.geometry[geometry].copy()
                mesh.apply_transform(transform)
                meshes.append(mesh)
            if not meshes:
                raise ValueError(f'No mesh geometry: {source}')
            if not any(getattr(m.visual, 'kind', None) == 'texture' for m in meshes):
                raise ValueError(f'No texture loaded for {source}; retain the original MTL and texture files')
            all_vertices = np.concatenate([m.vertices for m in meshes])
            low, high = all_vertices[:, 1].min(), all_vertices[:, 1].max()
            height = high - low
            if not np.isfinite(height) or height <= 0:
                raise ValueError(f'Invalid mesh height for {subject}')
            for mesh in meshes:
                vertices = mesh.vertices.copy() / height
                vertices[:, 1] -= low / height + .5
                mesh.vertices = vertices
            yaw = 0.
            if args.smplx_root:
                # Pickle is only for trusted fittings obtained through the data provider.
                with (args.smplx_root / subject / 'smplx_param.pkl').open('rb') as handle:
                    fitting = pickle.load(handle)
                yaw = float(np.asarray(fitting['global_orient']).reshape(-1, 3)[0, 1]) * 180 / math.pi
            scene = pyrender.Scene(bg_color=[1., 1., 1., 1.], ambient_light=[.4, .4, .4])
            for mesh in meshes:
                scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
            for angle in (0., 90., 180., 270.):
                scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.), pose=camera_pose(angle))
            camera = pyrender.IntrinsicsCamera(focal, focal, size / 2, size / 2, znear=.5, zfar=2.5)
            node = scene.add(camera, pose=camera_pose(yaw))
            for folder in folders.values():
                folder.mkdir(parents=True, exist_ok=True)
            for view in range(views):
                pose = camera_pose(yaw + view * 30.)
                scene.set_pose(node, pose)
                color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                mask = (depth > 0).astype(np.uint8) * 255
                if not mask.any():
                    raise ValueError(f'Empty rendering: subject {subject}, view {view}')
                Image.fromarray(color[:, :, :3]).save(folders['img'] / f'{view:02d}.jpg', quality=95)
                Image.fromarray(mask).save(folders['mask'] / f'{view:02d}.png')
                # pyrender OpenGL c2w -> dataset OpenCV world-to-camera.
                extrinsic = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(pose)
                np.save(folders['parm'] / f'{view:02d}_extrinsic.npy', extrinsic[:3])
                np.save(folders['parm'] / f'{view:02d}_intrinsic.npy', intrinsic)
            metadata.write_text(json.dumps({'implementation': 'approximate pyrender preprocessing',
                'resolution': size, 'fovy_degrees': fovy, 'radius': radius, 'views': views,
                'yaw_degrees': yaw, 'normalization': 'height=1; y centered; x/z scaled without recentering',
                'original_height': float(height)}, indent=2))
            print(f'Rendered {subject}: 12 RGB images, masks and camera pairs', flush=True)
    finally:
        renderer.delete()


if __name__ == '__main__':
    main()
