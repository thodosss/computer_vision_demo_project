"""Image-to-image search: encode a query image with CLIP and search the same index."""
import logging
from pathlib import Path
from typing import List, Tuple

from .image_processor import load_image, validate_image
from .semantic_search import SemanticSearch

log = logging.getLogger("ovs.image_search")


class ImageSearch:
    def __init__(self, searcher: SemanticSearch):
        self.searcher = searcher

    def search_image(self, image_path: Path, k: int = 5, exclude_self: bool = True) -> List[Tuple[dict, float]]:
        """Return the k most similar indexed images.

        If the query image is itself in the index, it is excluded by default (it would always
        be rank 1 with similarity ~1.0).
        """
        image_path = Path(image_path)
        ok, err = validate_image(image_path)
        if not ok:
            raise ValueError(f"Query image {image_path} is not a valid image: {err}")
        emb = self.searcher.embedder.encode_images([load_image(image_path, self.searcher.config.max_image_side)])[0]
        query_path = str(image_path.resolve())
        results = self.searcher.search_embedding(emb, k + 1 if exclude_self else k)
        if exclude_self:
            results = [(rec, s) for rec, s in results if rec["path"] != query_path][:k]
        return results
