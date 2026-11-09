"""Open-vocabulary image search: single entry point.

    python main.py index    [--input data/images] [--batch-size 1]
    python main.py search   --query "a dog sitting on grass" [--top-k 5] [--detect]
    python main.py search   --image-query data/images/000000000285.jpg
    python main.py classify --image data/images/000000000285.jpg --labels "car,dog,bicycle,bear,person"
    python main.py detect   --image data/images/000000001584.jpg [--conf 0.25]

Only one neural network is loaded on the GPU at a time (CLIP or YOLO), never both.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402
from app.utils import Timer, setup_logging  # noqa: E402


def cmd_classify(argv) -> int:
    from app.embedding_generator import ClipEmbedder
    from app.zero_shot_classifier import ZeroShotClassifier

    cfg = Config()
    ap = argparse.ArgumentParser(prog="main.py classify", description="Zero-shot classification with CLIP")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--labels", required=True, help='comma-separated, e.g. "car,dog,bicycle"')
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=cfg.device)
    args = ap.parse_args(argv)
    setup_logging(cfg.log_dir / "classify.log")
    cfg.device = args.device

    embedder = ClipEmbedder(cfg)
    try:
        with Timer() as t:
            scores = ZeroShotClassifier(embedder).classify(args.image, args.labels.split(","))
        print(f"\nImage: {args.image}   ({embedder.device}, {t.elapsed * 1000:.0f} ms)\n")
        print(f"{'label':<25} {'similarity':>10} {'relative':>9}")
        for s in scores:
            bar = "#" * round(s.relative_score * 30)
            print(f"{s.label:<25} {s.similarity:>10.4f} {s.relative_score:>9.3f}  {bar}")
        print("\nsimilarity = cosine similarity (not a probability).\n"
              "relative   = softmax over *these* labels only; it always sums to 1, even if none of the\n"
              "             labels is in the image, so it is not a calibrated probability.")
    finally:
        embedder.close()
    return 0


def cmd_detect(argv) -> int:
    from app.object_detector import ObjectDetector

    cfg = Config()
    ap = argparse.ArgumentParser(prog="main.py detect", description="Object detection with YOLOv8n")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=cfg.device)
    args = ap.parse_args(argv)
    setup_logging(cfg.log_dir / "detect.log")
    cfg.device = args.device

    detector = ObjectDetector(cfg, conf=args.conf)
    try:
        with Timer() as t:
            dets, out = detector.detect_and_save(args.image)
        print(f"\nImage: {args.image}   ({detector.device}, {t.elapsed * 1000:.0f} ms)\n")
        print(f"{'object':<16} {'conf':>5}  box (x1, y1, x2, y2)")
        for d in dets:
            print(f"{d.label:<16} {d.confidence:>5.2f}  {tuple(round(v) for v in d.box)}")
        if not dets:
            print("(no objects above the confidence threshold)")
        print(f"\nAnnotated image: {out}")
    finally:
        detector.close()
    return 0


def main() -> int:
    commands = {"index", "search", "classify", "detect"}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__)
        return 0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "index":
        from scripts import index_images
        return index_images.main(argv)
    if cmd == "search":
        from scripts import search
        return search.main(argv)
    if cmd == "classify":
        return cmd_classify(argv)
    return cmd_detect(argv)


if __name__ == "__main__":
    sys.exit(main())
