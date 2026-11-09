"""Local Gradio dashboard for the open-vocabulary search pipeline.

    python main.py ui            # http://127.0.0.1:7860

Binds to 127.0.0.1 only and never creates a public share link.
Only one network sits on the GPU at a time: CLIP stays loaded for fast searches and is
unloaded before YOLO runs (and vice versa).
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

from .config import Config  # noqa: E402
from .image_processor import list_images  # noqa: E402
from .image_search import ImageSearch  # noqa: E402
from .semantic_search import SemanticSearch  # noqa: E402
from .utils import Timer, gpu_memory_mb, ram_usage_mb, release_gpu_memory  # noqa: E402
from .vector_database import VectorDatabase  # noqa: E402
from .zero_shot_classifier import ZeroShotClassifier  # noqa: E402

log = logging.getLogger("ovs.ui")

SCORE_NOTE = ("Scores are CLIP cosine similarities (not probabilities). On this collection a good match "
              "usually scores ≥ 0.27; if the best result is ~0.24 or lower, there is probably no real match.")


class Backend:
    """Owns the models and makes sure CLIP and YOLO are never on the GPU together."""

    def __init__(self, config: Config):
        self.config = config
        self.lock = threading.Lock()  # Gradio runs handlers in worker threads
        self.searcher: Optional[SemanticSearch] = None
        self.detector = None
        self._load_index()

    # -- model management -----------------------------------------------------------
    def _load_index(self) -> None:
        try:
            db = VectorDatabase.load(self.config.index_dir)
        except FileNotFoundError:
            self.searcher = None
            return
        old = self.searcher
        self.searcher = SemanticSearch(self.config, db=db, embedder=old._embedder if old else None)

    def _clip(self) -> SemanticSearch:
        if self.searcher is None:
            raise gr.Error("No index found. Open the 'Index & status' tab and click 'Re-index images'.")
        if self.detector is not None:
            self.detector.close()
            self.detector = None
            release_gpu_memory()
        _ = self.searcher.embedder  # loads CLIP if needed
        return self.searcher

    def _yolo(self):
        if self.searcher is not None:
            self.searcher.close()  # unload CLIP first
        if self.detector is None:
            from .object_detector import ObjectDetector
            self.detector = ObjectDetector(self.config)
        return self.detector

    def _unload_all(self) -> None:
        if self.searcher is not None:
            self.searcher.close()
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        release_gpu_memory()

    # -- helpers ----------------------------------------------------------------------------
    @staticmethod
    def _label_text(rec) -> str:
        label = rec.get("label")
        return ", ".join(label) if isinstance(label, list) else (label or "-")

    def _format(self, results, elapsed_ms: float, device: str):
        gallery = [(rec["path"], f"#{i} · {s:.3f} · {self._label_text(rec)[:40]}")
                   for i, (rec, s) in enumerate(results, 1)]
        rows = [[i, round(s, 4), rec["filename"], self._label_text(rec)] for i, (rec, s) in enumerate(results, 1)]
        best = results[0][1] if results else 0.0
        warn = " ⚠️ **Low best score: probably no real match in the collection.**" if best < 0.25 else ""
        status = f"{len(results)} results in {elapsed_ms:.0f} ms on **{device}**.{warn}\n\n{SCORE_NOTE}"
        return gallery, rows, status

    # -- handlers ---------------------------------------------------------------------------
    def text_search(self, query: str, k: int):
        if not query or not query.strip():
            raise gr.Error("Type a description first, e.g. 'a person riding a bicycle'.")
        with self.lock:
            s = self._clip()
            with Timer() as t:
                results = s.search_text(query.strip(), int(k))
            return self._format(results, t.elapsed * 1000, s.embedder.device)

    def image_search(self, image_path: Optional[str], k: int, include_self: bool):
        if not image_path:
            raise gr.Error("Upload an image or pick one from the collection.")
        with self.lock:
            s = self._clip()
            with Timer() as t:
                try:
                    results = ImageSearch(s).search_image(Path(image_path), int(k), exclude_self=not include_self)
                except ValueError as exc:
                    raise gr.Error(str(exc))
            return self._format(results, t.elapsed * 1000, s.embedder.device)

    def classify(self, image_path: Optional[str], labels: str):
        if not image_path:
            raise gr.Error("Upload an image or pick one from the collection.")
        with self.lock:
            s = self._clip()
            try:
                with Timer() as t:
                    scores = ZeroShotClassifier(s.embedder).classify(Path(image_path), labels.split(","))
            except ValueError as exc:
                raise gr.Error(str(exc))
            label_dict = {x.label: x.relative_score for x in scores}
            rows = [[x.label, round(x.similarity, 4), round(x.relative_score, 4)] for x in scores]
            best = scores[0].similarity
            warn = (" ⚠️ **Highest similarity is low: possibly none of these labels is in the image**, "
                    "even though one label still gets a high relative score." if best < 0.22 else "")
            note = (f"{t.elapsed * 1000:.0f} ms on **{s.embedder.device}**.{warn}\n\n"
                    "*similarity* = cosine similarity. *relative* = softmax over **your** labels only: it always "
                    "sums to 1, so it is not a calibrated probability.")
            return label_dict, rows, note

    def detect(self, image_path: Optional[str], conf: float):
        if not image_path:
            raise gr.Error("Upload an image or pick one from the collection.")
        with self.lock:
            det = self._yolo()
            det.conf = float(conf)
            with Timer() as t:
                dets, out = det.detect_and_save(Path(image_path))
            rows = [[d.label, round(d.confidence, 3), *[round(v) for v in d.box]] for d in dets]
            note = f"{len(dets)} objects in {t.elapsed * 1000:.0f} ms on **{det.device}** (YOLOv8n, conf ≥ {conf:.2f})."
            return str(out), rows, note

    def reindex(self, progress=gr.Progress()):
        with self.lock:
            self._unload_all()
            progress(0.1, desc="Indexing images with CLIP (batch size 1)...")
            from scripts import index_images
            rc = index_images.main([])
            self._load_index()
            stats_path = self.config.results_dir / "index_stats.json"
            stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        if rc != 0:
            raise gr.Error("Indexing failed; see results/logs/index.log")
        return (f"Indexed **{stats.get('images_indexed')}** images in {stats.get('total_indexing_s')} s "
                f"({stats.get('images_per_s')} img/s on {stats.get('device')}).\n\n" + self.status())

    def status(self) -> str:
        n = len(self.searcher.db) if self.searcher else 0
        clip = self.searcher is not None and self.searcher._embedder is not None
        lines = [
            f"- **Indexed images:** {n}  (`{self.config.index_dir}`)",
            f"- **Model:** {self.config.model_name} (FP32, batch size 1)",
            f"- **Loaded now:** {'CLIP' if clip else ''}{'YOLOv8n' if self.detector else ''}"
            f"{'' if clip or self.detector else 'nothing'}",
            f"- **Process RAM:** {ram_usage_mb():.0f} MB",
        ]
        g = gpu_memory_mb()
        if g:
            lines.append(f"- **GPU:** this app {g['allocated_mb']:.0f} MB allocated (peak {g['peak_allocated_mb']:.0f} MB); "
                         f"device free {g['device_free_mb']:.0f} / {g['device_total_mb']:.0f} MB")
        else:
            lines.append("- **GPU:** CUDA not available, running on CPU")
        return "\n".join(lines)


def build_ui(config: Optional[Config] = None) -> gr.Blocks:
    config = config or Config()
    backend = Backend(config)
    collection: List[str] = [str(p) for p in list_images(config.images_dir)]
    by_name = {Path(p).name: p for p in collection}
    names = sorted(by_name)

    def pick(name):
        return by_name.get(name)

    def ex(name):  # example image path
        return by_name.get(name)

    with gr.Blocks(title="Open-Vocabulary Image Search") as ui:
        gr.Markdown("# Open-Vocabulary Image Search\nCLIP ViT-B/32 + FAISS on a local image collection · "
                    "runs fully offline on this machine")

        with gr.Tab("Text search"):
            with gr.Row():
                q = gr.Textbox(label="Describe the image you are looking for", scale=4,
                               placeholder="a person riding a bicycle")
                k = gr.Slider(1, 30, value=8, step=1, label="Top-K", scale=1)
            btn = gr.Button("Search", variant="primary")
            gr.Examples([["a person riding a bicycle"], ["a dog sitting on grass"], ["a red double-decker bus"],
                         ["a computer monitor on a desk"], ["food on a plate"], ["a building at night"]],
                        inputs=[q], label="Example queries")
            gal = gr.Gallery(label="Results", columns=4, object_fit="contain")
            info = gr.Markdown()
            tbl = gr.Dataframe(headers=["rank", "score", "file", "COCO labels"], label="Details")
            btn.click(backend.text_search, [q, k], [gal, tbl, info])
            q.submit(backend.text_search, [q, k], [gal, tbl, info])

        with gr.Tab("Image search"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_q = gr.Image(type="filepath", label="Query image (upload, paste or pick below)")
                    pick_q = gr.Dropdown(names, label="…or pick from the collection")
                    k2 = gr.Slider(1, 30, value=8, step=1, label="Top-K")
                    self_q = gr.Checkbox(label="Include the query image itself if it is indexed")
                    btn2 = gr.Button("Find similar", variant="primary")
                with gr.Column(scale=3):
                    gal2 = gr.Gallery(label="Similar images", columns=4, object_fit="contain")
                    info2 = gr.Markdown()
                    tbl2 = gr.Dataframe(headers=["rank", "score", "file", "COCO labels"], label="Details")
            pick_q.change(pick, pick_q, img_q)
            btn2.click(backend.image_search, [img_q, k2, self_q], [gal2, tbl2, info2])

        with gr.Tab("Zero-shot classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_c = gr.Image(type="filepath", label="Image")
                    pick_c = gr.Dropdown(names, label="…or pick from the collection")
                    labels = gr.Textbox(label="Candidate labels (comma-separated)",
                                        value="car, dog, bicycle, building, person, bear, cat, food")
                    btn3 = gr.Button("Classify", variant="primary")
                with gr.Column(scale=2):
                    lab = gr.Label(label="Relative score (softmax over your labels)", num_top_classes=8)
                    info3 = gr.Markdown()
                    tbl3 = gr.Dataframe(headers=["label", "similarity", "relative"], label="Scores")
            examples_c = [[ex(n), l] for n, l in [
                ("000000000285.jpg", "bear, dog, cat, person"),
                ("000000039769.jpg", "car, bicycle, building"),
                ("000000001584.jpg", "bus, tram, truck, car, train")] if ex(n)]
            if examples_c:
                gr.Examples(examples_c, inputs=[img_c, labels], label="Examples (the 2nd has no correct label)")
            pick_c.change(pick, pick_c, img_c)
            btn3.click(backend.classify, [img_c, labels], [lab, tbl3, info3])

        with gr.Tab("Object detection (YOLOv8n)"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_d = gr.Image(type="filepath", label="Image")
                    pick_d = gr.Dropdown(names, label="…or pick from the collection")
                    conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
                    btn4 = gr.Button("Detect", variant="primary")
                with gr.Column(scale=2):
                    out_d = gr.Image(label="Detections")
                    info4 = gr.Markdown()
                    tbl4 = gr.Dataframe(headers=["object", "conf", "x1", "y1", "x2", "y2"], label="Objects")
            pick_d.change(pick, pick_d, img_d)
            btn4.click(backend.detect, [img_d, conf], [out_d, tbl4, info4])

        with gr.Tab("Index & status"):
            st = gr.Markdown(backend.status())
            with gr.Row():
                gr.Button("Refresh status").click(backend.status, None, st)
                gr.Button("Re-index images", variant="secondary").click(backend.reindex, None, st)
            gr.Markdown(f"Images are read from `{config.images_dir}`. Add or remove files there, then click "
                        "**Re-index images** (rebuilds the index from scratch; ~8 s for 300 images on the GTX 1060).")
    return ui


def launch(port: int = 7860, device: Optional[str] = None) -> None:
    cfg = Config()
    if device:
        cfg.device = device
    ui = build_ui(cfg)
    ui.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=port, share=False, inbrowser=False,
        allowed_paths=[str(cfg.images_dir), str(cfg.results_dir)])
