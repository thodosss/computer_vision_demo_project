"""CLIP image/text embeddings -> L2-normalized float32 numpy arrays."""
import logging
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .config import Config
from .model_loader import load_clip, unload_model
from .utils import release_gpu_memory, select_device

log = logging.getLogger("ovs.embed")


def _features(output) -> torch.Tensor:
    # transformers 5.x returns BaseModelOutputWithPooling whose pooler_output is the
    # projected embedding; 4.x returned the tensor directly.
    return output if torch.is_tensor(output) else output.pooler_output


class ClipEmbedder:
    def __init__(self, config: Optional[Config] = None, device: Optional[str] = None):
        self.config = config or Config()
        wanted = select_device(device or self.config.device)
        self.model, self.processor, self.device = load_clip(
            self.config.model_name, wanted, self.config.models_dir)

    # -- public API ---------------------------------------------------------
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt")
        return self._run(lambda m, x: m.get_image_features(**x), inputs)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        return self._run(lambda m, x: m.get_text_features(**x), inputs)

    def close(self) -> None:
        if getattr(self, "model", None) is not None:
            unload_model(self.model)
            self.model = None
            release_gpu_memory()

    # -- internals ----------------------------------------------------------
    def _run(self, fn, inputs) -> np.ndarray:
        try:
            return self._forward(fn, inputs)
        except torch.cuda.OutOfMemoryError:
            if self.device != "cuda":
                raise
            # CLIP's input size is fixed (224x224), so lowering resolution is not an option here;
            # batch size is already 1 by default, so the remaining remedy is the CPU.
            log.error(
                "CUDA out of memory during CLIP inference (batch=%d). Likely cause: VRAM held by "
                "other processes (desktop, browser) or a second model. Moving CLIP to CPU.",
                inputs["pixel_values"].shape[0] if "pixel_values" in inputs else -1,
            )
            release_gpu_memory()
            self.model.to("cpu")
            self.device = "cpu"
            release_gpu_memory()
            return self._forward(fn, inputs)

    def _forward(self, fn, inputs) -> np.ndarray:
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = _features(fn(self.model, inputs))
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy().astype(np.float32)
