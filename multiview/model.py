"""MV-Adapter pipeline construction shared by training and inference."""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = '4277e0018232bac82bb2c103caf0893cedb711be'
BASE_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'
VAE_MODEL = 'madebyollin/sdxl-vae-fp16-fix'
ADAPTER_MODEL = 'huanngzh/mv-adapter'
PROMPT = 'a full body photograph of a person, white background'


def build_pipeline(device='cuda', dtype=torch.bfloat16):
    source = ROOT / 'third_party' / 'MV-Adapter'
    if not source.is_dir():
        raise FileNotFoundError('Run bash scripts/setup_multiview.sh first')
    sys.path.insert(0, str(source))
    from diffusers import AutoencoderKL, DDPMScheduler
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

    local_only = os.environ.get('HF_HUB_OFFLINE', '0') == '1'
    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=dtype, local_files_only=local_only)
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
        BASE_MODEL, vae=vae, torch_dtype=dtype, variant='fp16', add_watermarker=False,
        local_files_only=local_only)
    pipe.init_custom_adapter(num_views=3, copy_attn_weights=False)
    pipe.load_custom_adapter(ADAPTER_MODEL, weight_name='mvadapter_i2mv_sdxl.safetensors',
                             local_files_only=local_only)
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode='interpolated', shift_scale=8.)
    # Preserve upstream's Euler inference sampler; DDPM is only the training
    # forward-noise process. Clipping SDXL latents during sampling is incorrect.
    pipe.training_scheduler = DDPMScheduler.from_config(pipe.scheduler.config, clip_sample=False)
    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype).requires_grad_(False).eval()
    for module in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        module.requires_grad_(False).eval()
    pipe.enable_vae_slicing()
    return pipe


@torch.no_grad()
def prompt_embeddings(pipe):
    positive, negative, pooled, negative_pooled = pipe.encode_prompt(
        PROMPT, device=pipe._execution_device, do_classifier_free_guidance=True, negative_prompt='')
    # Cache the fixed prompt. Keep pipeline modules on one device because
    # diffusers derives its inference execution device from those modules.
    return {'prompt_embeds': positive, 'negative_prompt_embeds': negative,
            'pooled_prompt_embeds': pooled, 'negative_pooled_prompt_embeds': negative_pooled}


@torch.no_grad()
def generate(pipe, embeddings, reference, control, seed=42, steps=30, guidance=3.):
    size = reference.shape[-1]
    return pipe(
        **embeddings, reference_image=reference.unsqueeze(0), control_image=control,
        height=size, width=size, num_images_per_prompt=3,
        num_inference_steps=steps, guidance_scale=guidance,
        generator=torch.Generator(device=pipe._execution_device).manual_seed(seed),
    ).images
