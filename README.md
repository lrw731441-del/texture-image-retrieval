# Texture Image Retrieval

Multi-scale texture-based image retrieval with local patch matching and template-matching rerank.

## Features

- **Global retrieval**: LBP + HOG + pHash + CNN multi-scale feature fusion (cosine similarity)
- **Local patch retrieval**: dense multi-scale LBP patch voting for partial/cropped queries
- **Template-matching rerank**: TM_CCOEFF_NORMED spatial verification on top candidates
- **Auto clothing crop**: face-detection-guided clothing region extraction for model-shot → fabric queries
- **Unicode path support**: all I/O handles Chinese/Unicode file paths
- **HEIC/AVIF fallback**: PIL-backed image loading when OpenCV codecs fail

## Quick Start

```bash
# 1. Install dependencies
pip install numpy opencv-python scikit-image torch torchvision matplotlib pillow

# 2. Build index (global + local patches)
python build_index.py --local_index

# 3. Search (global mode)
python search.py path/to/query.jpg

# 4. Search (local patch mode — best for crop/detail queries)
python search.py path/to/crop.jpg --local_mode --rerank_template

# 5. Search (model photo → fabric, with auto clothing crop)
python search.py path/to/model.jpg --local_mode --rerank_template --auto_crop -o result.png
```

## Command Reference

| Scenario | Command |
|----------|---------|
| Full image → similar images | `python search.py image.jpg` |
| Fabric crop → full fabric | `python search.py crop.jpg --local_mode --rerank_template` |
| Model photo → fabric | `python search.py model.jpg --local_mode --rerank_template --auto_crop` |
| Save visualization | add `-o result.png` |
| Adjust recall | `--rerank_factor 20` (default: 5) |

## Files

| File | Purpose |
|------|---------|
| `utils.py` | Feature extraction, template matching, clothing crop, image I/O |
| `build_index.py` | Batch index builder (global + local patches) |
| `search.py` | Search CLI (global / local / rerank / auto-crop modes) |
| `feature_extractor.py` | Legacy LBP+HOG extractor |
| `generate_test_images.py` | Synthetic test image generator |

## License

MIT
