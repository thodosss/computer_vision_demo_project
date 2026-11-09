"""Logging, device selection and memory monitoring helpers."""
import gc
import logging
import time
from pathlib import Path
from typing import Optional

import psutil
import torch

log = logging.getLogger("ovs")


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> None:
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    # transformers/httpx are chatty at INFO
    for noisy in ("httpx", "httpx2", "urllib3", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def select_device(preference: str = "auto") -> str:
    if preference == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        log.warning("CUDA requested but not available; falling back to CPU")
    return "cpu"


def gpu_memory_mb() -> dict:
    """Current/peak memory allocated by this process's torch allocator, plus device free/total."""
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    mb = 2 ** 20
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / mb, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / mb, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / mb, 1),
        "device_free_mb": round(free / mb, 1),
        "device_total_mb": round(total / mb, 1),
    }


def ram_usage_mb() -> float:
    """Resident set size of the current process."""
    return round(psutil.Process().memory_info().rss / 2 ** 20, 1)


def release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Timer:
    """with Timer() as t: ...;  t.elapsed  (seconds)"""

    def __enter__(self):
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        return False
