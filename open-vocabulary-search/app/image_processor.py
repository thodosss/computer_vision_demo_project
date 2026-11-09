"""Image discovery, validation, memory-friendly loading and result contact sheets."""
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("ovs.images")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: Path) -> List[Path]:
    directory = Path(directory)
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def validate_image(path: Path) -> Tuple[bool, Optional[str]]:
    """Check that a file is a decodable image without keeping it in memory."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True, None
    except Exception as exc:  # PIL raises many exception types for corrupt files
        return False, f"{type(exc).__name__}: {exc}"


def image_metadata(path: Path) -> dict:
    path = Path(path)
    with Image.open(path) as img:
        width, height = img.size
    return {
        "filename": path.name,
        "path": str(path.resolve()),
        "width": width,
        "height": height,
        "file_size": path.stat().st_size,
    }


def load_image(path: Path, max_side: int = 384) -> Image.Image:
    """Open an image as RGB, downscaled so its longest side is <= max_side.

    For JPEGs, draft() lets libjpeg decode directly at a reduced scale, so the
    full-resolution bitmap never needs to sit in RAM.
    """
    img = Image.open(path)
    if img.format == "JPEG":
        img.draft("RGB", (max_side, max_side))
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.BICUBIC)
    return img


def make_contact_sheet(
    items: Sequence[Tuple[Path, str]],
    out_path: Path,
    title: str = "",
    thumb: int = 224,
    cols: int = 5,
) -> Path:
    """Save a grid of thumbnails. items = [(image_path, caption), ...]."""
    font = ImageFont.load_default()
    pad, caption_h = 8, 30
    title_h = 30 if title else 0
    rows = max(1, (len(items) + cols - 1) // cols)
    n_cols = min(cols, max(1, len(items)))
    sheet = Image.new(
        "RGB",
        (pad + n_cols * (thumb + pad), title_h + pad + rows * (thumb + caption_h + pad)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((pad, 8), title[:150], fill="black", font=font)
    for i, (path, caption) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = title_h + pad + r * (thumb + caption_h + pad)
        try:
            im = load_image(path, max_side=thumb)
            sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
        except Exception as exc:
            draw.text((x, y), f"cannot open: {exc}"[:40], fill="red", font=font)
        draw.text((x, y + thumb + 2), caption[:38], fill="black", font=font)
        draw.text((x, y + thumb + 15), Path(path).name[:38], fill="gray", font=font)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
