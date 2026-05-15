"""Batch index builder — scans images/ and saves features + metadata to data/.

Two modes (backward compatible):
  1. Global  — full-image LBP+HOG+CNN descriptor  (default, always built)
  2. Local   — dense multi-scale patch descriptors (--local_index)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from utils import (
    extract_features,
    load_image,
    extract_dense_patches,
    extract_local_texture,
    preprocess_image,
    to_gray,
    augment_fabric,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATCH_STRIDE,
    DEFAULT_PATCH_SCALES,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGES_DIR = Path(__file__).resolve().parent / "images"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p


def build_global_index(images: list):
    """Build global feature index (always runs).  Returns metadata list."""
    features_list = []
    meta = []

    print(f"\n[global] Building global index for {len(images)} images ...")
    t0 = time.time()

    for i, img_path in enumerate(images, 1):
        rel = img_path.relative_to(IMAGES_DIR).as_posix()
        try:
            img = load_image(str(img_path))
            feat = extract_features(img)
            features_list.append(feat)
            meta.append({"index": len(meta), "path": rel, "stem": img_path.stem})
            print(f"  [{i:4d}/{len(images)}] {rel}")
        except Exception as e:
            print(f"  [SKIP] {rel} — {e}")

    if not features_list:
        print("[ERROR] No valid images processed for global index.")
        return None

    features = np.stack(features_list, axis=0).astype(np.float32)

    np.save(str(DATA_DIR / "features.npy"), features)
    with open(DATA_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"[global] Done. {len(features)} images in {elapsed:.1f}s  "
          f"({features.shape[1]} dims)")
    print(f"  → {DATA_DIR / 'features.npy'}")
    print(f"  → {DATA_DIR / 'index.json'}")
    return meta


def build_local_index(images: list, global_meta: list,
                      patch_size: int, stride: int, scales: list,
                      augment_n: int = 0):
    """Build dense multi-scale local patch feature index.

    When augment_n > 0, each image also contributes N augmented variants
    (perspective + elastic + gamma) so 3-D draped queries can match
    against flat 2-D fabric scans.
    """
    patch_feats = []
    patch_meta = []
    total_patches = 0

    print(f"\n[local] Building local patch index for {len(images)} images ...")
    print(f"        patch_size={patch_size}  stride={stride}  "
          f"scales={scales}")
    if augment_n > 0:
        print(f"        augment: {augment_n} variants per image "
              f"(×{augment_n + 1} total)")
    t0 = time.time()

    for i, img_path in enumerate(images, 1):
        rel = img_path.relative_to(IMAGES_DIR).as_posix()
        try:
            img = load_image(str(img_path))
            processed = preprocess_image(img)

            # Collect all source images: original + augmented variants
            sources = [(processed, "original")]
            if augment_n > 0:
                aug_variants = augment_fabric(processed, n_variants=augment_n,
                                              seed=i)
                for vi, va in enumerate(aug_variants):
                    sources.append((va, f"aug-{vi + 1}"))

            n_total = 0
            for src_img, tag in sources:
                gray = to_gray(src_img)
                patches = extract_dense_patches(gray, scales=scales,
                                                patch_size=patch_size,
                                                stride=stride)
                for patch_arr, (x, y, w, h), scale in patches:
                    feat = extract_local_texture(patch_arr)
                    patch_feats.append(feat)
                    patch_meta.append({
                        "image_idx": i - 1,
                        "x": x, "y": y, "w": w, "h": h,
                        "scale": round(scale, 2),
                    })
                n_total += len(patches)

            total_patches += n_total
            tag_str = f"×{augment_n + 1}" if augment_n > 0 else ""
            print(f"  [{i:4d}/{len(images)}] {rel}  → {n_total} patches"
                  f"{'  ' + tag_str if tag_str else ''}")

        except Exception as e:
            print(f"  [SKIP] {rel} — {e}")

    if not patch_feats:
        print("[ERROR] No patches extracted.")
        return

    features = np.stack(patch_feats, axis=0).astype(np.float32)
    np.save(str(DATA_DIR / "features_local.npy"), features)
    with open(DATA_DIR / "local_meta.json", "w", encoding="utf-8") as f:
        json.dump(patch_meta, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"[local]  Done. {total_patches} patches from {len(images)} images "
          f"in {elapsed:.1f}s  ({features.shape[1]} dims per patch)")
    print(f"  → {DATA_DIR / 'features_local.npy'}")
    print(f"  → {DATA_DIR / 'local_meta.json'}")


def build_index(args):
    images = list(iter_images(IMAGES_DIR))
    if not images:
        print("[ERROR] No images found in images/ folder.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Global index — always built
    global_meta = build_global_index(images)
    if global_meta is None:
        return

    # Local index — optional
    if args.local_index:
        scales = [float(s) for s in args.patch_scales.split(",")]
        build_local_index(images, global_meta,
                          patch_size=args.patch_size,
                          stride=args.patch_stride,
                          scales=scales,
                          augment_n=args.augment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build image feature index (global + optional local patches)"
    )
    parser.add_argument(
        "--images", type=str, default=None,
        help="Override images directory (default: images/)"
    )
    parser.add_argument(
        "--local_index", action="store_true", default=False,
        help=(
            "Also build a dense multi-scale local-patch LBP index "
            "for patch-level retrieval (saved as features_local.npy + "
            "local_meta.json)."
        ),
    )
    parser.add_argument(
        "--patch_size", type=int, default=DEFAULT_PATCH_SIZE,
        help=f"Square patch side length in pixels (default: {DEFAULT_PATCH_SIZE})"
    )
    parser.add_argument(
        "--patch_stride", type=int, default=DEFAULT_PATCH_STRIDE,
        help=f"Stride between adjacent patches in pixels (default: {DEFAULT_PATCH_STRIDE})"
    )
    parser.add_argument(
        "--patch_scales", type=str,
        default=",".join(str(s) for s in DEFAULT_PATCH_SCALES),
        help="Comma-separated scale factors for multi-scale sampling "
             "(default: 1.0,0.7,0.5)"
    )
    parser.add_argument(
        "--augment", type=int, default=0, metavar="N",
        help=(
            "Generate N augmented variants per image (perspective + elastic "
            "+ gamma) so 3-D draped/model queries can match flat fabric "
            "scans.  Each image contributes (N+1)× patches to the local "
            "index.  Recommended: 4."
        ),
    )
    args = parser.parse_args()

    if args.images:
        IMAGES_DIR = Path(args.images).resolve()

    build_index(args)
