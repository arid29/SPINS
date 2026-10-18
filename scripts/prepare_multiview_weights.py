"""Download the public weights used by the single-image multiview trainer."""
import os
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1] / 'pretrained' / 'huggingface'))

from huggingface_hub import hf_hub_download, snapshot_download


def main():
    snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', allow_patterns=[
        'model_index.json', 'scheduler/*', 'tokenizer/*', 'tokenizer_2/*',
        'text_encoder/config.json', 'text_encoder/model.fp16.safetensors',
        'text_encoder_2/config.json', 'text_encoder_2/model.fp16.safetensors',
        'unet/config.json', 'unet/diffusion_pytorch_model.fp16.safetensors'], max_workers=4)
    snapshot_download('madebyollin/sdxl-vae-fp16-fix', allow_patterns=[
        'config.json', 'diffusion_pytorch_model.safetensors'], max_workers=2)
    hf_hub_download('huanngzh/mv-adapter', 'mvadapter_i2mv_sdxl.safetensors')
    print('Multiview weights ready.')


if __name__ == '__main__':
    main()
