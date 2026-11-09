"""FAISS IndexFlatIP (exact cosine on normalized vectors) + SQLite metadata.

Row i of the FAISS index corresponds to images.id == i in SQLite.
"""
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import faiss
import numpy as np

log = logging.getLogger("ovs.db")

INDEX_FILE = "images.faiss"
DB_FILE = "metadata.sqlite"

_COLUMNS = ("filename", "path", "width", "height", "file_size", "label")


class VectorDatabase:
    def __init__(self, dim: int = 512, model_name: str = ""):
        self.dim = dim
        self.model_name = model_name
        self.index = faiss.IndexFlatIP(dim)
        self.records: List[dict] = []

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, records: Sequence[dict]) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) embeddings, got {embeddings.shape}")
        if len(records) != len(embeddings):
            raise ValueError("embeddings and records length mismatch")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("embeddings must be L2-normalized for cosine similarity")
        start = len(self.records)
        for i, rec in enumerate(records):
            self.records.append({**rec, "id": start + i})
        self.index.add(embeddings)

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[dict, float]]:
        query = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        k = min(k, len(self))
        if k <= 0:
            return []
        scores, ids = self.index.search(query, k)
        return [(self.records[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def get_vector(self, idx: int) -> np.ndarray:
        return self.index.reconstruct(int(idx))

    # -- persistence --------------------------------------------------------
    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        tmp_index = index_dir / (INDEX_FILE + ".tmp")
        tmp_db = index_dir / (DB_FILE + ".tmp")
        faiss.write_index(self.index, str(tmp_index))
        if tmp_db.exists():
            tmp_db.unlink()
        con = sqlite3.connect(tmp_db)
        try:
            con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            con.execute(
                "CREATE TABLE images (id INTEGER PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL, "
                "width INTEGER, height INTEGER, file_size INTEGER, label TEXT)")
            con.executemany("INSERT INTO meta VALUES (?, ?)", [
                ("dim", str(self.dim)), ("model_name", self.model_name), ("count", str(len(self)))])
            con.executemany(
                "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(r["id"], *(_encode(r.get(c)) for c in _COLUMNS)) for r in self.records])
            con.commit()
        finally:
            con.close()
        # Swap in both files only after both were written successfully.
        os.replace(tmp_index, index_dir / INDEX_FILE)
        os.replace(tmp_db, index_dir / DB_FILE)
        log.info("Saved %d vectors to %s", len(self), index_dir)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorDatabase":
        index_dir = Path(index_dir)
        index_path, db_path = index_dir / INDEX_FILE, index_dir / DB_FILE
        if not index_path.exists() or not db_path.exists():
            raise FileNotFoundError(f"No index found in {index_dir}. Run scripts/index_images.py first.")
        con = sqlite3.connect(db_path)
        try:
            meta = dict(con.execute("SELECT key, value FROM meta"))
            rows = con.execute(f"SELECT id, {', '.join(_COLUMNS)} FROM images ORDER BY id").fetchall()
        finally:
            con.close()
        db = cls(dim=int(meta["dim"]), model_name=meta.get("model_name", ""))
        db.index = faiss.read_index(str(index_path))
        db.records = [{"id": r[0], **{c: _decode(c, v) for c, v in zip(_COLUMNS, r[1:])}} for r in rows]
        if db.index.ntotal != len(db.records):
            raise RuntimeError(
                f"Index/metadata mismatch: {db.index.ntotal} vectors vs {len(db.records)} rows. Re-index.")
        return db


def _encode(value):
    return json.dumps(value) if isinstance(value, (list, dict)) else value


def _decode(column, value):
    if column == "label" and isinstance(value, str) and value.startswith("["):
        return json.loads(value)
    return value
