import numpy as np
import pytest

from app.config import Config
from app.image_processor import image_metadata, load_image
from app.image_search import ImageSearch
from app.semantic_search import SemanticSearch
from app.vector_database import VectorDatabase
from app.zero_shot_classifier import ZeroShotClassifier


@pytest.fixture
def colour_search(embedder, color_images):
    names = ["red", "green", "blue"]
    emb = np.vstack([embedder.encode_images([load_image(color_images[n])]) for n in names])
    db = VectorDatabase(dim=512, model_name=Config().model_name)
    db.add(emb, [{**image_metadata(color_images[n]), "label": [n]} for n in names])
    return SemanticSearch(Config(), embedder=embedder, db=db)


# -- image-to-image search ---------------------------------------------------------
def test_image_search_excludes_query_itself(colour_search, color_images):
    results = ImageSearch(colour_search).search_image(color_images["red"], k=2)
    assert len(results) == 2
    assert all(rec["label"] != ["red"] for rec, _ in results)


def test_image_search_include_self_ranks_query_first(colour_search, color_images):
    rec, score = ImageSearch(colour_search).search_image(color_images["red"], k=1, exclude_self=False)[0]
    assert rec["label"] == ["red"]
    assert score == pytest.approx(1.0, abs=1e-3)


def test_image_search_finds_similar_colour(colour_search, color_images, tmp_path):
    from PIL import Image
    query = tmp_path / "dark_red.png"
    Image.new("RGB", (300, 300), (160, 10, 10)).save(query)
    rec, _ = ImageSearch(colour_search).search_image(query, k=1)[0]
    assert rec["label"] == ["red"]


def test_image_search_rejects_invalid_image(colour_search, color_images):
    with pytest.raises(ValueError):
        ImageSearch(colour_search).search_image(color_images["corrupt"], k=1)


# -- zero-shot classification -----------------------------------------------------
def test_zero_shot_picks_correct_colour(embedder, color_images):
    scores = ZeroShotClassifier(embedder).classify(color_images["blue"], ["red", "green", "blue"])
    assert scores[0].label == "blue"
    assert sum(s.relative_score for s in scores) == pytest.approx(1.0, abs=1e-5)
    assert all(-1.0 <= s.similarity <= 1.0 for s in scores)


def test_zero_shot_label_embeddings_normalized(embedder):
    emb = ZeroShotClassifier(embedder).label_embeddings(["car", "dog"])
    assert emb.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_zero_shot_requires_two_labels(embedder, color_images):
    with pytest.raises(ValueError):
        ZeroShotClassifier(embedder).classify(color_images["red"], ["red", " "])


# -- optional YOLO stage ------------------------------------------------------------
BUS = Config().images_dir / "000000001584.jpg"


@pytest.mark.skipif(not BUS.exists(), reason="COCO sample image not downloaded")
def test_yolo_detects_bus(tmp_path):
    pytest.importorskip("ultralytics")
    from app.object_detector import ObjectDetector
    det = ObjectDetector(Config())
    try:
        dets, out = det.detect_and_save(BUS, out_dir=tmp_path)
    finally:
        det.close()
    assert out.exists()
    assert any(d.label == "bus" and d.confidence > 0.5 for d in dets)
    x1, y1, x2, y2 = dets[0].box
    assert x2 > x1 and y2 > y1
