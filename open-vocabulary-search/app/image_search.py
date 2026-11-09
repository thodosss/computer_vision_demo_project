"""Image-to-image search: encode a query image with CLIP and search the same index."""
import logging
from pathlib import Path
from typing import List, Tuple

from .image_processor import load_image, validate_image
from .semantic_search import SemanticSearch

log = logging.getLogger("ovs.image_search")

# Cosine similarity at or above which an indexed image is treated as the query image itself.
# Path matching alone is not enough: the web UI hands us a temporary copy of the file.
DUPLICATE_THRESHOLD = 0.995


class ImageSearch:
    def __init__(self, searcher: SemanticSearch):
        self.searcher = searcher

    def search_image(self, image_path: Path, k: int = 5, exclude_self: bool = True) -> List[Tuple[dict, float]]:
        """Return the k most similar indexed images.

        With exclude_self (default), the query image is removed from the results if it is in the index,
        recognised by path or by a near-identical embedding (otherwise it would always be rank 1 with
        similarity ~1.0).
        """
        image_path = Path(image_path)
        ok, err = validate_image(image_path)
        if not ok:
            raise ValueError(f"Query image {image_path} is not a valid image: {err}")
        emb = self.searcher.embedder.encode_images([load_image(image_path, self.searcher.config.max_image_side)])[0]
        if not exclude_self:
            return self.searcher.search_embedding(emb, k)
        query_path = str(image_path.resolve())
        results = self.searcher.search_embedding(emb, k + 3)  # room for a few duplicates
        kept = [(rec, s) for rec, s in results if rec["path"] != query_path and s < DUPLICATE_THRESHOLD]
        if len(kept) < len(results):
            log.info("Excluded %d result(s) identical to the query image", len(results) - len(kept))
        return kept[:k]
