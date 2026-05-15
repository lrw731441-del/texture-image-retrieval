# Texture Image Retrieval Engine

Production-grade multi-scale texture-based reverse image search system for fabric and material libraries. Supports global descriptor retrieval, dense local-patch voting, and template-matching spatial verification with automatic clothing-region extraction for cross-domain (model-shot → flat-fabric) queries.

## Architecture

```
Query Image
    │
    ├─ Global Branch (LBP + HOG + pHash + CNN)
    │     └─ Cosine similarity against full-image index
    │
    ├─ Local Branch (dense multi-scale LBP patches)
    │     ├─ Adaptive patch sizing per query resolution
    │     ├─ Per-patch nearest-neighbour voting
    │     └─ Source-image score aggregation
    │
    └─ Re-rank (TM_CCOEFF_NORMED template matching)
          └─ Spatial consistency verification on top-N candidates
```

## Key Features

| Module | Description |
|--------|-------------|
| **Global Descriptor** | Fused LBP (R=1,2,3 × 3 scales) + spatial-pyramid HOG + DCT pHash + dual-layer ResNet-18 CNN with multi-scale grid pooling. Group-weighted L2 normalisation. |
| **Local Patch Index** | Dense multi-scale patch sampling (~29 patches/image @ 48 px, 3 scales). Compact 54-dim LBP descriptor per patch. Adaptive query-side patch sizing for cross-resolution matching. |
| **Template-Matching Re-rank** | Normalised cross-correlation (`TM_CCOEFF_NORMED`) with automatic query-to-candidate scale adjustment. Spatial match region visualised as bounding box. |
| **Auto Clothing Crop** | Haar-cascade face detection + anthropometric torso estimation. Zero extra dependencies. Fallback centre-crop heuristic. |
| **I/O Robustness** | Unicode path support via `np.frombuffer` + `cv2.imdecode`. PIL fallback for HEIC/AVIF and other formats unsupported by OpenCV. |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build feature index (global + local patches)
python build_index.py --local_index

# 3. Search
python search.py path/to/query.jpg                           # global vector search
python search.py path/to/crop.jpg --local_mode --rerank_template   # local-patch search
python search.py path/to/model.jpg --local_mode --rerank_template --auto_crop  # model→fabric
```

## CLI Reference

```bash
python search.py <query> [OPTIONS]

Search Modes:
  (default)                 Global descriptor retrieval (cosine similarity)
  --local_mode              Dense multi-scale local-patch voting
  --rerank_template         Two-stage retrieval with TM_CCOEFF_NORMED re-rank
  --auto_crop               Automatic clothing-region extraction for model photos

Options:
  --top_k N                 Number of results (default: 5)
  --threshold FLOAT         Minimum similarity threshold (default: 0.0)
  --rerank_factor N         Coarse-recall multiplier for re-rank (default: 5)
  --patch_size N            Square patch side in pixels (default: 48)
  --patch_stride N          Stride between patches (default: 24)
  --top_per_patch N         DB patches considered per query patch (default: 3)
  -o, --output PATH         Save visualisation to file
  -h, --help                Show full help
```

## Dependencies

```
numpy>=1.24.0
opencv-python>=4.8.0
scikit-image>=0.21.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
Pillow
```

## Project Structure

```
.
├── utils.py                     # Core library (features, matching, I/O, crop)
├── build_index.py               # Offline index builder
├── search.py                    # Online search CLI
├── feature_extractor.py         # Standalone LBP+HOG extractor (legacy)
├── generate_test_images.py      # Synthetic test-image generator
├── requirements.txt
├── data/                        # Generated index files (gitignored)
├── images/                      # Image library (gitignored)
└── .gitignore
```

## License

MIT
