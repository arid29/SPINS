"""Prepare local VGG/LPIPS weights and an explicitly supplied or public LGM model."""
import argparse
import shutil
from pathlib import Path

import torch
import torchvision
import lpips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lgm-checkpoint', type=Path, help='Copy an existing LGM checkpoint')
    group.add_argument('--download-public', action='store_true', help='Download the public LGM checkpoint')
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / 'pretrained'
    output.mkdir(exist_ok=True)
    target = output / 'model.safetensors'
    if not target.exists():
        if args.download_public:
            torch.hub.download_url_to_file(
                'https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors', str(target))
        else:
            shutil.copyfile(args.lgm_checkpoint, target)
    else:
        print(f'Keeping existing {target}; remove explicitly to replace it.')
    vgg = output / 'vgg16_features.pth'
    if not vgg.exists():
        network = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        torch.save(network.features.state_dict(), vgg)
    lpips_path = output / 'lpips_vgg.pth'
    if not lpips_path.exists():
        source = Path(lpips.__file__).parent / 'weights' / 'v0.1' / 'vgg.pth'
        shutil.copyfile(source, lpips_path)
    print(f'Prepared weights in {output}')


if __name__ == '__main__':
    main()
