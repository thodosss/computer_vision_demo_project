import numpy as np
import pytest

from app.config import Config
from app.image_processor import image_metadata, list_images, load_image, validate_image
from app.semantic_search import SemanticSearch
from app.vector_database import VectorDatabase


# -- image loading -------------------------------------------------------------
def test_list_and_validate_images(color_images, tmp_path):
    found = list_images(tmp_path)
    assert len(found) == 4
    assert validate_image(color_images["red"]) == (True, None)
    ok, err = validate_image(color_images["corrupt"])
    assert not ok and err


def test_load_image_downscales_and_converts_rgb(color_images):
    img = load_image(color_images["red"], max_side=224)
    assert img.mode == "RGB"
    assert max(img.size) <= 224
    meta = image_metadata(color_images["red"])
    assert (meta["width"], meta["height"]) == (640, 480)
    assert meta["file_size"] > 0


# -- embeddings ---------------------------------------------------------------
def test_image_embedding_dimension_and_norm(embedder, color_images):
    emb = embedder.encode_images([load_image(color_images["red"])])
    assert emb.shape == (1, 512)
    assert emb.dtype == np.float32
    assert np.linalg.norm(emb[0]) == pytest.approx(1.0, abs=1e-4)


def test_text_embedding_dimension_and_norm(embedder):
    emb = embedder.encode_texts(["a photo of a dog", "a building at night"])
    assert emb.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)


# -- FAISS + metadata ---------------------------------------------------------------
def _build_db(embedder, color_images):
    names = ["red", "green", "blue"]
    emb = np.vstack([embedder.encode_images([load_image(color_images[n])]) for n in names])
    db = VectorDatabase(dim=512, model_name=Config().model_name)
    db.add(emb, [{**image_metadata(color_images[n]), "label": [n]} for n in names])
    return db


def test_faiss_returns_expected_number_of_results(embedder, color_images):
    db = _build_db(embedder, color_images)
    q = embedder.encode_texts(["something"])[0]
    assert len(db.search(q, k=2)) == 2
    assert len(db.search(q, k=10)) == 3  # capped at index size
    scores = [s for _, s in db.search(q, k=3)]
    assert scores == sorted(scores, reverse=True)


def test_rejects_unnormalized_embeddings():
    db = VectorDatabase(dim=4)
    with pytest.raises(ValueError):
        db.add(np.full((1, 4), 3.0, dtype=np.float32), [{"filename": "x", "path": "x"}])


def test_metadata_persistence_roundtrip(embedder, color_images, tmp_path):
    db = _build_db(embedder, color_images)
    db.save(tmp_path / "index")
    loaded = VectorDatabase.load(tmp_path / "index")
    assert len(loaded) == 3
    assert loaded.records == db.records
    assert loaded.model_name == Config().model_name
    np.testing.assert_allclose(loaded.get_vector(1), db.get_vector(1))


# -- text-to-image search -------------------------------------------------------
@pytest.mark.parametrize("colour", ["red", "green", "blue"])
def test_text_search_finds_matching_colour(embedder, color_images, colour):
    search = SemanticSearch(Config(), embedder=embedder, db=_build_db(embedder, color_images))
    top_rec, _ = search.search_text(f"a solid {colour} colored image", k=1)[0]
    assert top_rec["label"] == [colour]
