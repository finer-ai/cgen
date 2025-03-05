import gc
import random
import numpy as np
import torch
from typing import Callable, Dict, Optional, Any
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    AutoencoderKL,
    StableDiffusionXLPipeline,
)
import logging

MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def seed_everything(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_scheduler(scheduler_config: Dict, name: str) -> Optional[Callable]:
    scheduler_factory_map = {
        "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True
        ),
        "DPM++ SDE Karras": lambda: DPMSolverSinglestepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True
        ),
        "DPM++ 2M SDE Karras": lambda: DPMSolverMultistepScheduler.from_config(
            scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        ),
        "Euler": lambda: EulerDiscreteScheduler.from_config(scheduler_config),
        "Euler a": lambda: EulerAncestralDiscreteScheduler.from_config(
            scheduler_config
        ),
        "DDIM": lambda: DDIMScheduler.from_config(scheduler_config),
    }
    return scheduler_factory_map.get(name, lambda: None)()


def free_memory() -> None:
    """Free up GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def preprocess_image_dimensions(width, height):
    if width % 8 != 0:
        width = width - (width % 8)
    if height % 8 != 0:
        height = height - (height % 8)
    return width, height

def load_pipeline(model_name: str, device: torch.device, vae: Optional[AutoencoderKL] = None) -> Any:
    """Load the Stable Diffusion pipeline."""
    try:
        pipeline = (
            StableDiffusionXLPipeline.from_single_file
            if model_name.endswith(".safetensors")
            else StableDiffusionXLPipeline.from_pretrained
        )

        pipe = pipeline(
            model_name,
            # vae=vae,
            torch_dtype=torch.float16,
            custom_pipeline="lpw_stable_diffusion_xl",
            use_safetensors=True,
            add_watermarker=False
        )
        pipe.to(device)
        return pipe
    except Exception as e:
        logging.error(f"Failed to load pipeline: {str(e)}", exc_info=True)
        raise