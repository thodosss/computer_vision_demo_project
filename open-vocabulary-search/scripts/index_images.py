"""Embed every image in a directory with CLIP and store the vectors in a FAISS index.

    python scripts/index_images.py --input data/images --batch-size 1

The index is rebuilt from scratch on every run and only replaces the old one when it finishes.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config  # noqa: E402
from app.embedding_generator import ClipEmbedder  # noqa: E402
from app.image_processor import image_metadata, list_images, load_image, validate_image  # noqa: E402
from app.utils import Timer, gpu_memory_mb, ram_usage_mb, setup_logging  # noqa: E402
from app.vector_database import VectorDatabase  # noqa: E402

log = logging.getLogger("ovs.index")


def load_labels(metadata_dir: Path) -> dict:
    """Optional ground-truth labels: data/metadata/labels.json = {filename: [label, ...]}."""
    path = metadata_dir / "labels.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=cfg.images_dir)
    ap.add_argument("--index-dir", type=Path, default=cfg.index_dir)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=cfg.device)
    ap.add_argument("--max-side", type=int, default=cfg.max_image_side,
                    help="downscale images to this longest side when loading")
    args = ap.parse_args()
    setup_logging(cfg.log_dir / "index.log")
    cfg.device, cfg.batch_size, cfg.max_image_side = args.device, args.batch_size, args.max_side

    paths = list_images(args.input)
    if not paths:
        log.error("No images found in %s", args.input)
        return 1
    log.info("Found %d candidate images in %s", len(paths), args.input)
    labels = load_labels(cfg.metadata_dir)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with Timer() as t_load:
        embedder = ClipEmbedder(cfg)
    log.info("Model load: %.2fs on %s | VRAM %s | RAM %.0f MB",
             t_load.elapsed, embedder.device, gpu_memory_mb(), ram_usage_mb())

    db = VectorDatabase(dim=cfg.embedding_dim, model_name=cfg.model_name)
    skipped, embed_time = [], 0.0
    batch_imgs, batch_recs = [], []

    def flush():
        nonlocal embed_time
        if not batch_imgs:
            return
        t0 = time.perf_counter()
        emb = embedder.encode_images(batch_imgs)
        embed_time += time.perf_counter() - t0
        db.add(emb, batch_recs)
        batch_imgs.clear()
        batch_recs.clear()

    with Timer() as t_total:
        for path in tqdm(paths, desc="Indexing", unit="img"):
            valid, err = validate_image(path)
            if not valid:
                log.warning("Skipping invalid image %s: %s", path, err)
                skipped.append(str(path))
                continue
            try:
                rec = image_metadata(path)
                img = load_image(path, cfg.max_image_side)
            except Exception as exc:
                log.warning("Skipping unreadable image %s: %s", path, exc)
                skipped.append(str(path))
                continue
            rec["label"] = labels.get(path.name)
            batch_imgs.append(img)
            batch_recs.append(rec)
            if len(batch_imgs) >= cfg.batch_size:
                flush()
        flush()
        db.save(args.index_dir)

    n = len(db)
    stats = {
        "images_indexed": n,
        "images_skipped": len(skipped),
        "device": embedder.device,
        "batch_size": cfg.batch_size,
        "model_load_s": round(t_load.elapsed, 3),
        "total_indexing_s": round(t_total.elapsed, 3),
        "embedding_s": round(embed_time, 3),
        "mean_embed_ms_per_image": round(1000 * embed_time / max(n, 1), 2),
        "images_per_s": round(n / t_total.elapsed, 2) if t_total.elapsed else None,
        "gpu": gpu_memory_mb(),
        "process_ram_mb": ram_usage_mb(),
    }
    embedder.close()
    log.info("Indexing stats: %s", json.dumps(stats))
    (cfg.results_dir / "index_stats.json").write_text(json.dumps(stats, indent=2))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
