"""Central configuration. Defaults favour stability on a GTX 1060 3GB: FP32, batch size 1."""
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    model_name: str = "openai/clip-vit-base-patch32"
    # "auto" picks CUDA when available, else CPU. Override with OVS_DEVICE=cpu|cuda.
    device: str = os.environ.get("OVS_DEVICE", "auto")
    batch_size: int = 1
    # Images are downscaled to this longest side when loaded (saves RAM / decode time).
    # CLIP's processor then resizes + center-crops to its fixed 224x224 input.
    max_image_side: int = 384
    embedding_dim: int = 512

    images_dir: Path = PROJECT_ROOT / "data" / "images"
    metadata_dir: Path = PROJECT_ROOT / "data" / "metadata"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    models_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"

    @property
    def log_dir(self) -> Path:
        return self.results_dir / "logs"
