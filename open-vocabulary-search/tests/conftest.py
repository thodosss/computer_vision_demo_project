import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def embedder():
    """One real CLIP instance for the whole test session (loading is the slow part)."""
    from app.embedding_generator import ClipEmbedder
    emb = ClipEmbedder(Config())
    yield emb
    emb.close()


@pytest.fixture
def color_images(tmp_path):
    """Three solid-colour images plus one corrupt file."""
    from PIL import Image
    paths = {}
    for name, rgb in {"red": (220, 20, 20), "green": (20, 180, 20), "blue": (20, 20, 220)}.items():
        p = tmp_path / f"{name}.jpg"
        Image.new("RGB", (640, 480), rgb).save(p)
        paths[name] = p
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"not an image at all")
    paths["corrupt"] = bad
    return paths
