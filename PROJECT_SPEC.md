# Project: Open Vocabulary & Semantic Search Computer Vision Pipeline

## 1. Role

Act as a Senior Computer Vision Engineer and Python Developer.

You have access to my local Ubuntu desktop. Your task is to inspect the environment, implement, test, and document a complete Computer Vision project.

Do not just explain how to build it. Actually create the project files, install the necessary dependencies where possible, execute the pipeline, and verify that it works.

Ask for my permission before performing destructive operations, modifying system-wide configurations, or downloading unusually large files.

---

## 2. Hardware and Environment

My desktop specifications:

- OS: Ubuntu 20.04.6 LTS, 64-bit
- CPU: Intel Core i5-6600, 4 cores
- GPU: NVIDIA GTX 1060, 3GB VRAM
- RAM: 16GB
- Display: 1920x1080

Hardware constraints:

- Start with FP32 inference.
- Use batch size 1.
- Use small pretrained models.
- Avoid large-scale model training.
- Avoid keeping multiple neural networks loaded simultaneously.
- Use CPU inference as a fallback.
- Monitor GPU memory usage.
- Prefer image resolution around 224–384 pixels.
- Release GPU memory when switching models.
- Do not assume that FP16 will improve performance on Pascal architecture.

Before implementation, inspect:

```bash
uname -a
lsb_release -a
lscpu
free -h
nvidia-smi
python3 --version
nvcc --version
```

Also inspect the installed NVIDIA driver and CUDA compatibility.

Do not change the existing NVIDIA driver or CUDA installation without asking first.

---

## 3. Main Objective

Build a local Computer Vision pipeline that allows me to:

1. Load a directory containing images.
2. Generate semantic embeddings for every image.
3. Store embeddings in a searchable vector database.
4. Search images using natural-language descriptions.
5. Search images using another image as a query.
6. Perform zero-shot classification using user-defined categories.
7. Display the most relevant results with similarity scores.
8. Optionally combine semantic search with object detection.

Example text queries:

- "A red car parked on the street"
- "A person riding a bicycle"
- "A dog sitting on grass"
- "A computer monitor on a desk"
- "A building at night"

The system must work completely locally after downloading the required models and datasets.

No external API calls should be necessary during inference.

---

## 4. Recommended Models and Technologies

Use the following lightweight stack as the starting point:

### Image and Text Embeddings

Model:

`openai/clip-vit-base-patch32`

Repository:

https://huggingface.co/openai/clip-vit-base-patch32

Use:

- Image encoder
- Text encoder
- Cosine similarity
- Normalized embeddings

Start with CLIP ViT-B/32.

Do not use larger CLIP variants unless benchmarking demonstrates that the hardware can handle them.

### Vector Search

Use FAISS CPU initially.

Repository:

https://github.com/facebookresearch/faiss

Use:

- IndexFlatIP for exact cosine similarity on normalized vectors.
- Persistent index storage.
- Image metadata stored separately in SQLite or JSON.

Do not use GPU FAISS initially.

### Optional Object Detection

Use:

`yolov8n`

Repository:

https://github.com/ultralytics/ultralytics

Use this only as an optional second stage.

The main semantic search pipeline must work without YOLO.

---

## 5. Dataset

Create a local image collection using a small subset of a public dataset.

Preferred option:

- COCO 2017 validation images, using only a small subset.
- Alternatively, use a small public image dataset with clear licensing.

Start with approximately 200–500 images.

Do not download the entire COCO dataset.

Create this structure:

```text
data/
    images/
    metadata/
```

Every image should have:

- Unique ID
- Filename
- Original path
- Image dimensions
- File size
- Optional ground-truth label

Avoid downloading duplicate images.

---

## 6. Project Structure

Create:

```text
open-vocabulary-search/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── model_loader.py
│   ├── image_processor.py
│   ├── embedding_generator.py
│   ├── vector_database.py
│   ├── semantic_search.py
│   ├── zero_shot_classifier.py
│   ├── image_search.py
│   └── utils.py
│
├── scripts/
│   ├── download_dataset.py
│   ├── index_images.py
│   ├── search.py
│   ├── benchmark.py
│   └── evaluate.py
│
├── data/
│   ├── images/
│   ├── metadata/
│   └── index/
│
├── results/
│
├── tests/
│
├── requirements.txt
├── README.md
└── main.py
```

Adapt the structure if necessary, but keep components modular.

---

## 7. Required Features

### Feature A: Image Indexing

Implement a script that:

1. Reads all images from `data/images/`.
2. Validates image files.
3. Loads CLIP.
4. Generates image embeddings.
5. Normalizes embeddings.
6. Stores them in FAISS.
7. Saves image metadata.
8. Displays progress and processing time.

Important:

Process one image at a time or use very small batches.

Do not load all original images into RAM.

Example:

```bash
python scripts/index_images.py --input data/images --batch-size 1
```

### Feature B: Text-to-Image Search

Implement natural-language search.

Example:

```bash
python scripts/search.py --query "a red car parked on a street" --top-k 5
```

The program should:

1. Encode the text query.
2. Normalize its embedding.
3. Search FAISS.
4. Return the top K images.
5. Display similarity scores.
6. Save a contact sheet of results.

### Feature C: Image-to-Image Search

Allow a user to provide an image as a query.

Example:

```bash
python scripts/search.py --image-query data/images/example.jpg --top-k 5
```

Return visually and semantically similar images.

### Feature D: Zero-Shot Classification

Allow arbitrary categories without retraining.

Example:

```bash
python main.py classify \
    --image data/images/example.jpg \
    --labels "car,dog,bicycle,building,person"
```

Return similarity scores for every candidate category.

Important:

Explain that CLIP scores are similarity scores, not calibrated probabilities.

### Feature E: Optional Object Detection

Implement an optional YOLOv8n integration.

The user should be able to run:

```bash
python main.py detect --image data/images/example.jpg
```

Return:

- Detected objects
- Bounding boxes
- Confidence scores
- Annotated output image

Do not load YOLO and CLIP simultaneously on the GPU unless memory testing confirms that it is safe.

---

## 8. Evaluation and Benchmarking

Implement a benchmark script.

Measure:

- Model loading time
- Image embedding latency
- Text embedding latency
- Total indexing time
- Search latency
- Images processed per second
- GPU VRAM consumption
- System RAM consumption

For semantic retrieval, implement:

- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)

If ground-truth labels are unavailable, clearly distinguish performance benchmarks from retrieval-quality evaluation.

Save results as:

```text
results/benchmark.json
results/benchmark.csv
```

Generate a simple performance report.

---

## 9. GPU Optimization

The system must prioritize stability over maximum speed.

Implement:

- Automatic CUDA/CPU device selection.
- FP32 as the default.
- Batch size 1 by default.
- Configurable image resolution.
- Inference mode using `torch.inference_mode()`.
- GPU memory monitoring when CUDA is available.
- Model unloading when necessary.

Use:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.inference_mode():
    # inference
    pass
```

Do not enable FP16 automatically.

If CUDA out-of-memory occurs:

1. Release model references where possible.
2. Call `torch.cuda.empty_cache()`.
3. Reduce image resolution.
4. Use CPU inference if necessary.
5. Explain the cause in the logs.

---

## 10. User Interface

Start with a command-line interface.

If the core pipeline works reliably, optionally implement a small local web interface using Gradio.

The interface should include:

- Image collection indexing
- Text search input
- Image query upload
- Top-K selection
- Zero-shot category input
- Result thumbnails
- Similarity scores

Run locally only.

Do not expose the web interface to the public internet.

---

## 11. Testing

Create tests for:

- Image loading
- Embedding dimensions
- Embedding normalization
- FAISS indexing
- Text-to-image search
- Image-to-image search
- Zero-shot classification
- Metadata persistence

Verify that:

```python
np.linalg.norm(embedding)
```

is approximately 1.0 for normalized embeddings.

Verify that FAISS returns the expected number of results.

---

## 12. Execution Plan

Work in phases.

### Phase 1 — Environment Inspection

Inspect the system and report:

- Python version
- NVIDIA driver
- CUDA availability
- PyTorch compatibility
- Available VRAM
- Disk space

Identify dependency conflicts before installing anything.

Prefer an isolated Python virtual environment.

### Phase 2 — Minimal Working Pipeline

Implement:

- CLIP model loading
- Image embeddings
- FAISS indexing
- Text-to-image search

Test with 10–20 images first.

### Phase 3 — Full Dataset

Expand to 200–500 images.

Verify that indexing and search work correctly.

### Phase 4 — Additional Features

Implement:

- Image-to-image search
- Zero-shot classification
- Optional YOLO integration

### Phase 5 — Evaluation

Run benchmarks and save metrics.

### Phase 6 — Documentation

Write a detailed README containing:

- Installation instructions
- Dataset acquisition
- Model information
- Commands
- Architecture
- Benchmark results
- Hardware limitations
- Troubleshooting

---

## 13. Important Engineering Rules

1. Do not blindly install the newest versions of all dependencies.
2. Check official documentation and compatibility before choosing versions.
3. Avoid changing system Python.
4. Use a virtual environment.
5. Do not download unnecessarily large models or datasets.
6. Do not start training unless explicitly requested.
7. Do not silently ignore errors.
8. Do not claim that a feature works unless it has actually been tested.
9. Save logs and benchmark outputs.
10. Keep the implementation modular and easy to extend.

If an installation fails, investigate the error and propose a compatible alternative instead of repeatedly attempting random package versions.

---

## 14. Final Deliverables

At the end, provide:

1. Complete working project.
2. Installation instructions.
3. Dataset download instructions.
4. Example commands for all features.
5. Benchmark results from my actual desktop.
6. VRAM and RAM usage.
7. Known limitations.
8. Suggested next experiments.

Do not finish after creating files.

Actually execute the minimal pipeline and verify that text-to-image search returns results.

The final report must clearly state:

- What was implemented.
- What was successfully tested.
- What did not work.
- How much VRAM and RAM were used.
- How fast the pipeline runs on the GTX 1060.
- Which command I should execute to start using it.

Start with Phase 1.

---

## 15. Implementation Decisions Log

### 2026-11-09 — Phase 1 findings and environment setup

- System `python3` is 3.8.10, too old for current torch/transformers/faiss. The project uses
  **Python 3.10.17** (`/usr/bin/python3.10`) in a project-local venv created with `uv`
  (`open-vocabulary-search/.venv`). System Python is untouched.
- **PyTorch pinned to `2.14.1+cu126`** (torchvision `0.29.1+cu126`). Reason: CUDA 12.8/12.9 wheels
  dropped Pascal (sm_61) in torch 2.8, and 2.14 is the last release that ships CUDA 12.6 wheels.
  Verified on the GTX 1060: `torch.cuda.is_available()` is True and CUDA matmul/conv ran
  (the wheel has sm_60 SASS, which runs on sm_61).
  **Do not upgrade torch past 2.14 or switch to the cu128+ index.**
- NVIDIA driver 570.211.01 (CUDA 12.8 capable) and the system CUDA toolkits (10.1 on PATH, 11.3, 12.2)
  were left unchanged; the pip wheels bundle their own CUDA runtime.
- transformers 5.x: `get_image_features` / `get_text_features` return `BaseModelOutputWithPooling`;
  use `.pooler_output` (already the 512-d projected embedding). Code must handle this.
- Measured: CLIP ViT-B/32 FP32 uses about 577 MiB VRAM after loading, with a peak of about 590 MiB at batch size 1.
- Baseline constraints: about 1 GB VRAM is used by the desktop and Firefox (about 2 GB free); about 4.5 GB RAM is free
  during normal desktop use.

### 2026-11-09 — Phase 2 (minimal pipeline) decisions

- **Offline model loading:** the official `openai/clip-vit-base-patch32` main branch only has
  `pytorch_model.bin`; with `use_safetensors=True` transformers pulls weights from a Hub conversion
  PR (`refs/pr/66`), which needs the network on every load. The first load now exports the model
  to `open-vocabulary-search/models/clip-vit-base-patch32/` (safetensors, 581 MB), and every later
  load reads only that folder. Verified with `HF_HUB_OFFLINE=1`.
- **Index layout:** `data/index/images.faiss` (IndexFlatIP) + `data/index/metadata.sqlite`.
  FAISS row i == `images.id` i. Both files are written to `.tmp` and swapped in only after both
  writes succeed. The index stores the model name, and search refuses an index built with a different model.
  Indexing currently rebuilds from scratch on each run (incremental indexing: not needed at ≤500 images).
- **"Configurable image resolution":** `--max-side` (default 384) controls how far images are downscaled
  on load (JPEG `draft()` decodes directly at reduced scale, saving RAM). CLIP's own input stays
  224×224 because ViT-B/32 positional embeddings are fixed, so resolution reduction is not
  an OOM remedy for CLIP; the OOM path falls back to CPU instead.
- **Optional labels:** `data/metadata/labels.json` (`{filename: [label, ...]}`) is read at indexing
  time if present (produced in Phase 3 from COCO annotations).

### 2026-11-09 — Phase 3 (full dataset) decisions

- Downloaded `annotations_trainval2017.zip` (241 MB, MD5 f4bbac642086de4f52a3fdda2de5fa2c) to
  `data/downloads/`; extracted only `instances_val2017.json` and `captions_val2017.json` to
  `data/downloads/coco_annotations/`.
- **Collection: 300 COCO val2017 images** (46 MB), including the 20 Phase 2 seed images. Selection is
  category-balanced by *primary* category (largest summed object area, which must cover ≥10% of the image,
  so the label is visually salient), round-robin over categories, seed 0 (reproducible).
  Result: 68 primary categories, 79/80 categories appear as some label, 2.57 labels/image on average.
  Small-object categories (knife, spoon, mouse, tie, ...) never dominate an image, so they only
  appear as secondary labels.
- Ground truth in `data/metadata/`: `labels.json` (categories ordered by area), `captions.json`
  (5 human captions/image), `dataset.json` (source URL, sha1, per-image Flickr CC license).
- **Evaluation plan (Phase 5):** COCO captions give text queries with exactly one known correct image,
  which makes them the primary Recall@K/MRR benchmark. Category labels support a secondary
  "label precision@K" metric.
