"""Download a small subset of COCO 2017 validation images, one file at a time.

Seed mode (Phase 2, no annotations needed):
    python scripts/download_dataset.py --count 20

Annotation mode (Phase 3): category-balanced subset + ground-truth labels and captions.
    python scripts/download_dataset.py --count 300 \
        --annotations data/downloads/coco_annotations/instances_val2017.json \
        --captions    data/downloads/coco_annotations/captions_val2017.json

Writes to data/metadata/:
    dataset.json   per-image source URL, sha1, COCO license
    labels.json    {filename: [category, ...]} ordered by total object area (largest first)
    captions.json  {filename: [caption, ...]} (5 human captions per image)

COCO images are licensed per-image under Flickr Creative Commons licenses (recorded in dataset.json);
annotations are CC BY 4.0. See https://cocodataset.org/#termsofuse.
"""
import argparse
import hashlib
import json
import logging
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config  # noqa: E402
from app.image_processor import validate_image  # noqa: E402
from app.utils import setup_logging  # noqa: E402

log = logging.getLogger("ovs.download")

COCO_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"

# A varied handful of COCO val2017 IDs (animals, vehicles, people, indoor scenes).
SEED_IDS = [
    139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
    1268, 1296, 1353, 1425, 1490, 1503, 1532, 1584, 1761, 39769,
]


def filename_for(coco_id: int) -> str:
    return f"{coco_id:012d}.jpg"


def download(url: str, dest: Path, retries: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "open-vocabulary-search/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 16):
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            log.warning("attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            time.sleep(2 * attempt)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"giving up on {url}")


# -- annotation-based selection ---------------------------------------------------
def load_coco(instances_path: Path):
    coco = json.loads(Path(instances_path).read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    licenses = {lic["id"]: {"name": lic["name"], "url": lic["url"]} for lic in coco["licenses"]}
    area = defaultdict(lambda: defaultdict(float))  # image_id -> category -> summed area
    for ann in coco["annotations"]:
        area[ann["image_id"]][cats[ann["category_id"]]] += ann["area"]
    labels = {
        img_id: [c for c, _ in sorted(per_cat.items(), key=lambda kv: -kv[1])]
        for img_id, per_cat in area.items()
    }
    primary_frac = {
        img_id: max(per_cat.values()) / (images[img_id]["width"] * images[img_id]["height"])
        for img_id, per_cat in area.items()
    }
    return cats, images, licenses, labels, primary_frac


def select_balanced(labels, primary_frac, categories, count, keep_ids, seed=0, min_primary_frac=0.10):
    """Pick ~count images so each category is the *dominant* object of a similar number of images.

    An image is a candidate for its primary (largest-area) category only if that category covers
    at least min_primary_frac of the image, so the ground-truth label is visually salient.
    """
    rng = random.Random(seed)
    selected = [i for i in keep_ids if i in labels]
    chosen = set(selected)
    by_primary = defaultdict(list)
    for img_id in sorted(labels):
        if primary_frac[img_id] >= min_primary_frac:
            by_primary[labels[img_id][0]].append(img_id)
    for pool in by_primary.values():
        rng.shuffle(pool)
    # Round-robin over categories until we reach the target.
    cat_names = sorted(categories.values())
    while len(selected) < count:
        progressed = False
        for cat in cat_names:
            pool = by_primary.get(cat, [])
            while pool and pool[-1] in chosen:
                pool.pop()
            if pool and len(selected) < count:
                img_id = pool.pop()
                selected.append(img_id)
                chosen.add(img_id)
                progressed = True
        if not progressed:
            break
    return selected


def load_captions(captions_path: Path) -> dict:
    data = json.loads(Path(captions_path).read_text())
    caps = defaultdict(list)
    for ann in data["annotations"]:
        caps[ann["image_id"]].append(ann["caption"].strip())
    return caps


# -- main ---------------------------------------------------------------------------
def main() -> int:
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=20, help="target number of images in the collection")
    ap.add_argument("--ids", type=int, nargs="*", help="explicit COCO val2017 image IDs (seed mode)")
    ap.add_argument("--annotations", type=Path, help="instances_val2017.json for balanced selection + labels")
    ap.add_argument("--captions", type=Path, help="captions_val2017.json (optional)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=cfg.images_dir)
    args = ap.parse_args()
    setup_logging(cfg.log_dir / "download.log")

    args.out.mkdir(parents=True, exist_ok=True)
    cfg.metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.metadata_dir / "dataset.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    coco_images, licenses = {}, {}
    if args.annotations:
        cats, coco_images, licenses, labels, primary_frac = load_coco(args.annotations)
        existing = [int(p.stem) for p in args.out.glob("*.jpg") if p.stem.isdigit()]
        ids = select_balanced(labels, primary_frac, cats, args.count, keep_ids=sorted(existing), seed=args.seed)
        log.info("Selected %d images (%d already on disk) covering %d primary categories",
                 len(ids), len(set(existing) & set(ids)), len({labels[i][0] for i in ids}))
    else:
        ids = (args.ids or SEED_IDS)[: args.count]

    seen_hashes = {m["sha1"]: name for name, m in manifest.items() if "sha1" in m}
    ok = failed = skipped = 0
    for coco_id in ids:
        name = filename_for(coco_id)
        dest = args.out / name
        url = COCO_URL.format(coco_id)
        if not (dest.exists() and validate_image(dest)[0]):
            try:
                download(url, dest)
            except RuntimeError as exc:
                log.error("%s", exc)
                failed += 1
                continue
            valid, err = validate_image(dest)
            if not valid:
                log.error("downloaded file %s is not a valid image (%s); removing", name, err)
                dest.unlink()
                failed += 1
                continue
            ok += 1
        else:
            skipped += 1
        sha1 = manifest.get(name, {}).get("sha1") or hashlib.sha1(dest.read_bytes()).hexdigest()
        if sha1 in seen_hashes and seen_hashes[sha1] != name:
            log.warning("%s duplicates %s; removing", name, seen_hashes[sha1])
            dest.unlink()
            continue
        seen_hashes[sha1] = name
        entry = {"coco_id": coco_id, "source_url": url, "sha1": sha1, "dataset": "COCO val2017"}
        if coco_id in coco_images:
            entry["license"] = licenses.get(coco_images[coco_id]["license"])
            entry["flickr_url"] = coco_images[coco_id].get("flickr_url")
        manifest[name] = entry
        if (ok + skipped) % 50 == 0:
            log.info("progress: %d/%d", ok + skipped, len(ids))

    manifest_path.write_text(json.dumps(manifest, indent=2))

    if args.annotations:
        on_disk = {p.name for p in args.out.glob("*.jpg")}
        gt = {filename_for(i): labels[i] for i in labels if filename_for(i) in on_disk}
        (cfg.metadata_dir / "labels.json").write_text(json.dumps(gt, indent=1, sort_keys=True))
        log.info("Wrote labels for %d images", len(gt))
        if args.captions:
            caps = load_captions(args.captions)
            cap_out = {filename_for(i): caps[i] for i in caps if filename_for(i) in on_disk}
            (cfg.metadata_dir / "captions.json").write_text(json.dumps(cap_out, indent=1, sort_keys=True))
            log.info("Wrote captions for %d images", len(cap_out))

    log.info("done: %d downloaded, %d already present, %d failed -> %s", ok, skipped, failed, args.out)
    return 1 if failed and not (ok or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
