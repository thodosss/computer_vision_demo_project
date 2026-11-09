"""Load / unload CLIP with OOM fallback to CPU.

The first load downloads from the Hugging Face Hub and exports a self-contained copy to
models/<name>/ (safetensors + processor files). Every later load reads only that folder,
so inference is fully offline. (The official openai repo ships only pytorch_model.bin on main;
transformers fetches safetensors from a Hub conversion PR, which cannot resolve offline.)
"""
import logging
from pathlib import Path
from typing import Tuple

import torch
from transformers import CLIPModel, CLIPProcessor

from .utils import release_gpu_memory

log = logging.getLogger("ovs.model")


def local_model_dir(models_dir: Path, model_name: str) -> Path:
    return Path(models_dir) / model_name.split("/")[-1]


def load_clip(model_name: str, device: str, models_dir: Path) -> Tuple[CLIPModel, CLIPProcessor, str]:
    """Load CLIP in FP32 (deliberately: FP16 gives little benefit on Pascal) in eval mode.

    Returns (model, processor, actual_device). If moving to the GPU runs out of memory,
    the model stays on the CPU and the returned device is "cpu".
    """
    local = local_model_dir(models_dir, model_name)
    if (local / "model.safetensors").exists():
        processor = CLIPProcessor.from_pretrained(local, local_files_only=True)
        model = CLIPModel.from_pretrained(local, dtype=torch.float32, local_files_only=True)
    else:
        log.info("%s not found locally; downloading from the Hugging Face Hub (one time)", model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True, dtype=torch.float32)
        local.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local)
        processor.save_pretrained(local)
        log.info("Exported %s to %s for offline use", model_name, local)
    model.eval()
    if device == "cuda":
        try:
            model.to("cuda")
        except torch.cuda.OutOfMemoryError:
            log.error(
                "CUDA OOM while moving %s to GPU (other processes may be holding VRAM; "
                "check nvidia-smi). Falling back to CPU.", model_name)
            model.to("cpu")
            release_gpu_memory()
            device = "cpu"
    log.info("Loaded %s on %s (fp32)", model_name, device)
    return model, processor, device


def unload_model(model) -> None:
    """Move a model off the GPU and free cached memory. The caller must drop its own references."""
    try:
        model.to("cpu")
    except Exception:  # pragma: no cover - best effort
        pass
    release_gpu_memory()
