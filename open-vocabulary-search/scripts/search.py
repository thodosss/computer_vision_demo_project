"""Search the indexed images with a natural-language query or with another image.

    python scripts/search.py --query "a red car parked on a street" --top-k 5
    python scripts/search.py --image-query data/images/000000000285.jpg --top-k 5
    python scripts/search.py --query "a dog on grass" --detect   # + YOLOv8n on the results
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config  # noqa: E402
from app.image_search import ImageSearch  # noqa: E402
from app.semantic_search import SemanticSearch  # noqa: E402
from app.utils import Timer, setup_logging  # noqa: E402

log = logging.getLogger("ovs.cli")


def print_results(results) -> None:
    print(f"\n{'rank':>4}  {'score':>7}  {'label':<30}  path")
    for rank, (rec, score) in enumerate(results, 1):
        label = rec.get("label")
        label = ", ".join(label) if isinstance(label, list) else (label or "-")
        print(f"{rank:>4}  {score:>7.4f}  {label[:30]:<30}  {rec['path']}")
    print("\nScores are cosine similarities between CLIP embeddings (not probabilities).")


def build_parser(cfg: Config) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--query", help="natural-language description")
    q.add_argument("--image-query", type=Path, help="path to a query image")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=cfg.device)
    ap.add_argument("--include-self", action="store_true",
                    help="image query: keep the query image itself in the results if it is indexed")
    ap.add_argument("--detect", action="store_true",
                    help="run YOLOv8n on the result images after CLIP is unloaded")
    ap.add_argument("--no-sheet", action="store_true", help="do not save a contact sheet")
    return ap


def main(argv=None) -> int:
    cfg = Config()
    args = build_parser(cfg).parse_args(argv)
    setup_logging(cfg.log_dir / "search.log")
    cfg.device = args.device

    searcher = SemanticSearch(cfg)
    try:
        with Timer() as t_load:
            device = searcher.embedder.device  # loads CLIP
        with Timer() as t:
            if args.query:
                title = f'"{args.query}"'
                results = searcher.search_text(args.query, args.top_k)
            else:
                title = f"similar to {args.image_query.name}"
                results = ImageSearch(searcher).search_image(
                    args.image_query, args.top_k, exclude_self=not args.include_self)
        log.info("%s top_k=%d device=%s | model load %.2fs | query %.1f ms (first call includes warm-up)",
                 title, args.top_k, device, t_load.elapsed, t.elapsed * 1000)
        print_results(results)
        if not args.no_sheet and results:
            out = searcher.save_contact_sheet(results, title)
            print(f"Contact sheet: {out}")
    finally:
        searcher.close()  # free CLIP before YOLO is loaded (sequential model loading)

    if args.detect and results:
        from app.object_detector import ObjectDetector
        detector = ObjectDetector(cfg)
        try:
            for rec, _ in results:
                dets, out = detector.detect_and_save(Path(rec["path"]))
                summary = ", ".join(f"{d.label} {d.confidence:.2f}" for d in dets) or "no objects"
                print(f"{rec['filename']}: {summary}  -> {out}")
        finally:
            detector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
