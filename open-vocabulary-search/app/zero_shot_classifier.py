"""Zero-shot classification with CLIP: arbitrary user-defined labels, no training.

For each label we build several prompts ("a photo of a {label}.", ...), average their normalized
text embeddings (prompt ensembling, as in the CLIP paper) and compare with the image embedding.

Two numbers are reported per label:
  * similarity: cosine similarity between image and label embeddings. Typical values are ~0.15-0.35.
    This is NOT a probability and is not comparable across differently-worded label sets.
  * relative score: softmax over the candidate labels of similarity * 100 (CLIP's learned logit scale).
    It sums to 1 over the labels *you* supplied, so it only says which candidate fits best relative to
    the others. It is not a calibrated probability: if none of the labels is present, one of them
    will still get a high relative score.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .embedding_generator import ClipEmbedder
from .image_processor import load_image, validate_image

DEFAULT_TEMPLATES = (
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
)
CLIP_LOGIT_SCALE = 100.0  # openai/clip-vit-base-patch32 learned logit_scale.exp()


@dataclass
class LabelScore:
    label: str
    similarity: float
    relative_score: float


class ZeroShotClassifier:
    def __init__(self, embedder: ClipEmbedder, templates: Sequence[str] = DEFAULT_TEMPLATES):
        self.embedder = embedder
        self.templates = tuple(templates)
        # CLIP's learned temperature (exp(logit_scale), 100.0 for ViT-B/32).
        self.logit_scale = float(embedder.model.logit_scale.exp().item()) if embedder.model is not None \
            else CLIP_LOGIT_SCALE

    def label_embeddings(self, labels: Sequence[str]) -> np.ndarray:
        out = []
        for label in labels:
            emb = self.embedder.encode_texts([t.format(label) for t in self.templates]).mean(axis=0)
            out.append(emb / np.linalg.norm(emb))
        return np.vstack(out).astype(np.float32)

    def classify_embedding(self, image_emb: np.ndarray, labels: Sequence[str]) -> List[LabelScore]:
        sims = self.label_embeddings(labels) @ image_emb
        logits = self.logit_scale * sims
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        scores = [LabelScore(l, float(s), float(p)) for l, s, p in zip(labels, sims, probs)]
        return sorted(scores, key=lambda x: -x.similarity)

    def classify(self, image_path: Path, labels: Sequence[str]) -> List[LabelScore]:
        labels = [l.strip() for l in labels if l.strip()]
        if len(labels) < 2:
            raise ValueError("Provide at least two candidate labels")
        ok, err = validate_image(Path(image_path))
        if not ok:
            raise ValueError(f"{image_path} is not a valid image: {err}")
        emb = self.embedder.encode_images([load_image(image_path, self.embedder.config.max_image_side)])[0]
        return self.classify_embedding(emb, labels)
