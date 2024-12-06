# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

from typing import AnyStr
import pathlib
from packaging import version

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.utils import is_torch_version, is_xformers_available


def init_sdxl_pipeline(scheduler: str = 'ddim',
                       device: torch.device = "cuda",
                       torch_dtype: torch.dtype = torch.float16,
                       variant: str = 'fp16',
                       local_files_only: bool = False,
                       force_download: bool = False,
                       torch_compile: bool = False,
                       enable_xformers: bool = False,
                       scaled_dot_product_attention: bool = True,
                       gradient_checkpoint: bool = False,
                       cpu_offload: bool = False,
                       vae_slicing: bool = False,
                       lora_path: AnyStr = None,
                       unet_path: AnyStr = None) -> StableDiffusionXLPipeline:
    """
    A tool for initial diffusers pipeline.

    Args:
        scheduler: any scheduler
        device: set device
        torch_dtype: data type
        variant: model variant
        local_files_only: prohibited download model
        force_download: forced download model
        torch_compile: use the `torch.compile` api to speed up unet
        enable_xformers: enable memory efficient attention from [xFormers]
        scaled_dot_product_attention: torch.nn.functional.scaled_dot_product_attention (SDPA)
                                      is an optimized and memory-efficient attention (similar to xFormers)
        gradient_checkpoint: activates gradient checkpointing for the current model
        cpu_offload: enable sequential cpu offload
        vae_slicing: enable sliced VAE decoding
        lora_path: load LoRA checkpoint
        unet_path: load unet checkpoint

    Returns:
            diffusers.StableDiffusionXLPipeline
    """

    # get model id
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    # vae
    vae_model_id = "madebyollin/sdxl-vae-fp16-fix"
    vae = AutoencoderKL.from_pretrained(vae_model_id,
                                        torch_dtype=torch.float16,
                                        local_files_only=local_files_only)
    print(f"Load {vae_model_id}")

    # process diffusion model
    if scheduler == 'ddim':
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch_dtype,
            variant=variant,
            local_files_only=local_files_only,
            force_download=force_download,
            scheduler=DDIMScheduler.from_pretrained(model_id,
                                                    subfolder="scheduler",
                                                    local_files_only=local_files_only,
                                                    force_download=force_download)
        ).to(device)
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            force_download=force_download,
        ).to(device)

    print(f"load diffusers pipeline: {model_id}")

    # process unet model if exist
    if unet_path is not None and pathlib.Path(unet_path).exists():
        print(f"=> load u-net from {unet_path}")
        pipeline.unet.from_pretrained(model_id, subfolder="unet")

    # process lora layers if exist
    if lora_path is not None and pathlib.Path(lora_path).exists():
        pipeline.unet.load_attn_procs(lora_path)
        print(f"=> load lora module from {lora_path} ...")

    # torch.compile
    if torch_compile:
        if is_torch_version(">=", "2.0.0"):
            # torch._inductor.config.conv_1x1_as_mm = True
            # torch._inductor.config.coordinate_descent_tuning = True
            # torch._inductor.config.epilogue_fusion = False
            # torch._inductor.config.coordinate_descent_check_all_directions = True

            pipeline.unet.to(memory_format=torch.channels_last)
            pipeline.vae.to(memory_format=torch.channels_last)

            # Compile the UNet and VAE.
            pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
            pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

            print(f"=> enable torch.compile")
        else:
            print(f"=> warning: calling torch.compile speed-up failed, since torch version <= 2.0.0")

    if scaled_dot_product_attention:
        pipeline.unet.set_attn_processor(AttnProcessor2_0())

    # Meta xformers
    if enable_xformers:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems during training, please update xFormers to at least 0.0.17. "
                    "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            print(f"=> enable xformers")
            pipeline.unet.enable_xformers_memory_efficient_attention()
        else:
            print(f"=> warning: xformers is not available.")

    # gradient checkpointing
    if gradient_checkpoint:
        # if pipeline.unet.is_gradient_checkpointing:
        if True:
            print(f"=> enable gradient checkpointing")
            pipeline.unet.enable_gradient_checkpointing()
        else:
            print("=> waring: gradient checkpointing is not activated for this model.")

    if cpu_offload:
        pipeline.enable_sequential_cpu_offload()

    if vae_slicing:
        pipeline.enable_vae_slicing()

    print(pipeline.scheduler)
    return pipeline
