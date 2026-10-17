# Third-party code

- `core/` contains adapted LGM components from [3DTopia/LGM](https://github.com/3DTopia/LGM), based on revision `fe8d12cff8c827df7bb77a3c8e8b37408cb6fe4c`. Its MIT notice is retained in the root `LICENSE`. The model, loader, local LPIPS weight loading, and camera schedule have been adapted for this approximate implementation.
- `core/attention.py` retains the Meta copyright header from [DINOv2](https://github.com/facebookresearch/dinov2). The Apache 2.0 license is included in `licenses/DINOv2-LICENSE`.
- `core/lpips.py` adapts the VGG perceptual-loss structure from [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity), loading weights from local files. Its upstream notice is included in `licenses/LPIPS-LICENSE`.
- `scripts/install_rasterizer.sh` fetches [ashawkey/diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization) at a pinned revision. It is an external dependency with its own license; its source and binaries are not vendored here.
- The optional renderer uses pyrender and trimesh as installed dependencies. The THuman scans, rendered data, and model weights are not redistributed.
- `assets/results/comparison_renderpeople.gif` shows model reconstructions of the [Renderpeople free sample](https://renderpeople.com/free-3d-people/) `rp_posed_00178_29`. The original mesh and textures are not redistributed. Source and rendering details are recorded in `assets/results/provenance_renderpeople.json`; the sample is used for inference only.

Paper-specific implementation and checkpoints are planned for a later update. No paper-level reproduction claim is made by this release.
