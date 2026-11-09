"""Search the indexed images with a natural-language query.

    python scripts/search.py --query "a red car parked on a street" --top-k 5
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config  # noqa: E402
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


def main() -> int:
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True, help="natural-language description")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=cfg.device)
    ap.add_argument("--no-sheet", action="store_true", help="do not save a contact sheet")
    args = ap.parse_args()
    setup_logging(cfg.log_dir / "search.log")
    cfg.device = args.device

    searcher = SemanticSearch(cfg)
    try:
        with Timer() as t_load:
            device = searcher.embedder.device  # loads CLIP
        with Timer() as t:
            results = searcher.search_text(args.query, args.top_k)
        log.info("query=%r top_k=%d device=%s | model load %.2fs | query %.1f ms "
                 "(text encoding + FAISS search; first call includes CUDA warm-up)",
                 args.query, args.top_k, device, t_load.elapsed, t.elapsed * 1000)
        print_results(results)
        if not args.no_sheet and results:
            out = searcher.save_contact_sheet(results, f'"{args.query}"')
            print(f"Contact sheet: {out}")
    finally:
        searcher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
