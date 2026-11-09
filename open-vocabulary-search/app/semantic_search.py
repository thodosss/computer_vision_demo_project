"""Text-to-image search over the persisted FAISS index."""
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import Config
from .embedding_generator import ClipEmbedder
from .image_processor import make_contact_sheet
from .vector_database import VectorDatabase

log = logging.getLogger("ovs.search")


class SemanticSearch:
    def __init__(self, config: Optional[Config] = None, embedder: Optional[ClipEmbedder] = None,
                 db: Optional[VectorDatabase] = None):
        self.config = config or Config()
        self.db = db or VectorDatabase.load(self.config.index_dir)
        if self.db.model_name and self.db.model_name != self.config.model_name:
            raise RuntimeError(
                f"Index was built with {self.db.model_name}, but config uses {self.config.model_name}. Re-index.")
        self._embedder = embedder

    @property
    def embedder(self) -> ClipEmbedder:
        if self._embedder is None:  # lazy: only load CLIP when a query needs encoding
            self._embedder = ClipEmbedder(self.config)
        return self._embedder

    def search_embedding(self, query: np.ndarray, k: int = 5) -> List[Tuple[dict, float]]:
        return self.db.search(query, k)

    def search_text(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        return self.search_embedding(self.embedder.encode_texts([query])[0], k)

    def save_contact_sheet(self, results: List[Tuple[dict, float]], title: str,
                           out_path: Optional[Path] = None) -> Path:
        if out_path is None:
            slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60] or "results"
            out_path = self.config.results_dir / "searches" / f"{slug}.png"
        items = [(Path(rec["path"]), f"#{rank} score={score:.3f}")
                 for rank, (rec, score) in enumerate(results, 1)]
        return make_contact_sheet(items, out_path, title=title)

    def close(self) -> None:
        if self._embedder is not None:
            self._embedder.close()
            self._embedder = None
