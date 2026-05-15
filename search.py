"""Search similar images using LBP+HOG+CNN multi-scale texture features.

Supports three search modes (all backward compatible):
  default        — global vector retrieval (cosine similarity)
  --local_mode   — dense multi-scale local-patch matching + source-image voting
  --rerank_template — template-matching re-rank on top of coarse results
  --local_mode --rerank_template — patch retrieval → template-matching fine rank
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from utils import (
    extract_features,
    load_image,
    template_match_score,
    extract_dense_patches,
    extract_local_texture,
    preprocess_image,
    to_gray,
    extract_clothing_region,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATCH_STRIDE,
    DEFAULT_PATCH_SCALES,
)

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGES_DIR = Path(__file__).resolve().parent / "images"


# ── Index loading ─────────────────────────────────────────────────────────

def load_global_index():
    features = np.load(str(DATA_DIR / "features.npy"))
    with open(DATA_DIR / "index.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return features, meta


def load_local_index():
    features = np.load(str(DATA_DIR / "features_local.npy"))
    with open(DATA_DIR / "local_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return features, meta


# ── Similarity helpers ────────────────────────────────────────────────────

def cosine_similarity(query_vec: np.ndarray, db_features: np.ndarray) -> np.ndarray:
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    db_norm = db_features / (np.linalg.norm(db_features, axis=1, keepdims=True) + 1e-12)
    return db_norm @ q_norm


def _load_rgb(path: str):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image
            pil_img = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _load_gray_shape(path: str):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            from PIL import Image
            img = np.array(Image.open(path).convert("L"))
        except Exception:
            raise ValueError(f"Cannot read image: {path}")
    return img.shape


def _template_dims_for_candidate(query_h, query_w, cand_h, cand_w):
    if query_h > cand_h or query_w > cand_w:
        scale = 0.8 * min(cand_h / query_h, cand_w / query_w)
        return (max(1, int(round(query_h * scale))),
                max(1, int(round(query_w * scale))))
    return (query_h, query_w)


# ── Local-patch retrieval ─────────────────────────────────────────────────

def search_local(query_patches_feats: np.ndarray,
                 db_feats: np.ndarray,
                 db_meta: list,
                 top_k: int,
                 top_per_patch: int = 3):
    """
    Match each query patch against the local-patch database and aggregate
    scores by source image.

    Args:
        query_patches_feats: (M, D) query patch features
        db_feats:            (N, D) database patch features
        db_meta:             list of {"image_idx": int, ...} per db patch
        top_k:               number of top-ranked source images to return
        top_per_patch:       top-N db patches considered per query patch

    Returns:
        List of (image_idx, aggregated_score) sorted descending.
    """
    M = query_patches_feats.shape[0]

    q_norm = query_patches_feats / (
        np.linalg.norm(query_patches_feats, axis=1, keepdims=True) + 1e-12
    )
    db_norm = db_feats / (
        np.linalg.norm(db_feats, axis=1, keepdims=True) + 1e-12
    )

    # All-pair similarities: (N, M)
    sims = db_norm @ q_norm.T

    # Per query patch, take top-per-patch matching db patches
    k = min(top_per_patch, sims.shape[0])
    top_rows = np.argpartition(-sims, k - 1, axis=0)[:k]  # (k, M)

    # Aggregate by source image
    scores = {}
    for j in range(M):
        for ki in range(k):
            db_idx = top_rows[ki, j]
            s = float(sims[db_idx, j])
            if s > 0:
                img_idx = db_meta[db_idx]["image_idx"]
                scores[img_idx] = scores.get(img_idx, 0.0) + s

    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return result[:top_k]


def do_local_search(args, global_meta):
    """Run local-patch retrieval with adaptive-scale query processing.

    For small queries (short side < 96 px) the image is kept at native
    resolution with proportionally scaled patch dimensions so texture
    scale stays consistent with the DB.  For larger queries the image is
    resized to 128×128 just like DB images were during indexing.
    """
    db_feats, db_patch_meta = load_local_index()

    print(f"Local index: {len(db_patch_meta)} patches across "
          f"{len(global_meta)} images, patch dim={db_feats.shape[1]}")

    query_img = load_image(args.query)

    # ── Auto clothing-region extraction (optional) ─────────────────────
    if getattr(args, "auto_crop", False):
        qh0, qw0 = query_img.shape[:2]
        query_img = extract_clothing_region(query_img)
        qh1, qw1 = query_img.shape[:2]
        if qh1 > 0 and qw1 > 0:
            print(f"Auto-crop: {qw0}×{qh0}  →  clothing region {qw1}×{qh1}")
        else:
            query_img = load_image(args.query)  # fallback to original
            print("Auto-crop: failed, using full image")

    gray = to_gray(query_img)
    qh, qw = gray.shape

    ref_side = 128.0
    small_threshold = 96  # below this px we treat the query as a crop/detail

    if min(qh, qw) >= small_threshold:
        # Large image (e.g. model photo) — keep native resolution for max
        # texture detail.  Patches are 48×48 px (same absolute size as DB
        # patches) so LBP descriptors are comparable.  Larger stride + extra
        # scales keep runtime manageable while ensuring clothing regions
        # get enough coverage.
        adaptive_ps = args.patch_size
        adaptive_stride = max(args.patch_stride,
                              int(round(min(qh, qw) / 20)))
        scales = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18]
        tag = f"native {qw}x{qh}"
    else:
        # Small crop/detail — adaptive patch sizing to preserve texture scale
        scale_factor = min(qh, qw) / ref_side
        adaptive_ps = max(16, int(round(args.patch_size * scale_factor)))
        adaptive_stride = max(8, int(round(args.patch_stride * scale_factor)))
        if scale_factor < 0.6:
            scales = [1.0, 0.8, 0.6, 0.4, 0.25]
        else:
            scales = [float(s) for s in args.patch_scales.split(",")]
        tag = f"adaptive sf={scale_factor:.2f}"

    patches = extract_dense_patches(gray, scales=scales,
                                    patch_size=adaptive_ps,
                                    stride=adaptive_stride)

    print(f"Query: {qw}x{qh}  →  {tag}  "
          f"ps={adaptive_ps} stride={adaptive_stride},  "
          f"{len(patches)} patches  scales={scales}")

    q_feats = np.stack(
        [extract_local_texture(p) for p, _, _ in patches], axis=0
    ).astype(np.float32)

    # Determine recall size
    recall_k = args.top_k * args.rerank_factor if args.rerank_template else args.top_k
    recall_k = min(recall_k, len(global_meta))

    top_images = search_local(q_feats, db_feats, db_patch_meta,
                              top_k=recall_k,
                              top_per_patch=args.top_per_patch)

    # Build coarse_results compatible with the downstream pipeline
    coarse_results = []
    for img_idx, score in top_images:
        coarse_results.append((img_idx, score, global_meta[img_idx]))

    return coarse_results, q_feats.shape[0]


# ── Visualization ─────────────────────────────────────────────────────────

def _setup_axes(n, cols):
    """Create a figure grid for n+1 images (query + n results)."""
    rows = 1 + (n + cols - 2) // (cols - 1) if n > 0 else 1
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.8 * cols, 2.8 * rows),
        gridspec_kw={"wspace": 0.25, "hspace": 0.35},
    )
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1)
    for rr in range(rows):
        for cc in range(cols):
            axes[rr, cc].axis("off")
    return fig, axes, rows, cols


def visualize_default(query_path: str, results: list,
                      title: str, output_path: str = None):
    """Global-mode visualisation — similarity bars only."""
    n = len(results)
    cols = min(n + 1, 5)
    fig, axes, rows, _cols = _setup_axes(n, cols)

    q_img = _load_rgb(query_path)
    if q_img is not None:
        axes[0, 0].imshow(q_img)
    axes[0, 0].set_title(f"Query\n{Path(query_path).name}", fontsize=9)

    for idx, (item_idx, score, meta_item) in enumerate(results):
        pos = idx + 1
        r = pos // _cols
        c = pos % _cols
        img_path = IMAGES_DIR / meta_item["path"]
        img = _load_rgb(str(img_path))
        if img is not None:
            axes[r, c].imshow(img)
        axes[r, c].set_title(
            f"#{pos}  sim={score:.4f}\n{meta_item['stem']}", fontsize=8
        )

    fig.suptitle(title, fontsize=13, y=0.98)
    _save_or_show(fig, output_path)


def visualize_rerank(query_path: str, results: list,
                     title: str, output_path: str = None):
    """Rerank-mode visualisation — match regions marked with red rectangles."""
    n = len(results)
    cols = min(n + 1, 5)
    fig, axes, rows, _cols = _setup_axes(n, cols)

    q_img = _load_rgb(query_path)
    if q_img is not None:
        axes[0, 0].imshow(q_img)
    axes[0, 0].set_title(f"Query\n{Path(query_path).name}", fontsize=9)

    q_h, q_w = _load_gray_shape(query_path)

    for idx, (item_idx, coarse_score, tm_score, location, meta_item) in enumerate(results):
        pos = idx + 1
        r = pos // _cols
        c = pos % _cols
        img_path = IMAGES_DIR / meta_item["path"]
        img = _load_rgb(str(img_path))
        if img is not None:
            axes[r, c].imshow(img)
            cand_h, cand_w = img.shape[:2]
            t_h, t_w = _template_dims_for_candidate(q_h, q_w, cand_h, cand_w)
            x, y = location
            rect = Rectangle((x, y), t_w, t_h,
                             fill=False, edgecolor="red", linewidth=2)
            axes[r, c].add_patch(rect)

        axes[r, c].set_title(
            f"#{pos}  coarse={coarse_score:.4f}  tm={tm_score:.4f}\n"
            f"{meta_item['stem']}", fontsize=7
        )

    fig.suptitle(title, fontsize=13, y=0.98)
    _save_or_show(fig, output_path)


def _save_or_show(fig, output_path):
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")
    else:
        try:
            plt.show()
        except Exception:
            print("[INFO] Display not available — use --output to save to file.")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Search similar images by texture features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Global vector search (default)
  python search.py query.jpg

  # Local patch matching
  python search.py query.jpg --local_mode

  # Global + template-matching rerank
  python search.py query.jpg --rerank_template

  # Local patch + template-matching rerank (best for local-detail queries)
  python search.py query.jpg --local_mode --rerank_template
        """,
    )
    parser.add_argument(
        "query", type=str, help="Path to query image"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Minimum similarity for coarse retrieval (default: 0.0)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save visualization to file instead of displaying"
    )

    # ── Rerank arguments ───────────────────────────────────────────────
    rerank = parser.add_argument_group("Template-matching rerank")
    rerank.add_argument(
        "--rerank_template", action="store_true", default=False,
        help=(
            "Enable two-stage retrieval: after coarse search, re-rank "
            "candidates via normalised cross-correlation (TM_CCOEFF_NORMED). "
            "The match region is drawn as a red rectangle in the visualisation."
        ),
    )
    rerank.add_argument(
        "--rerank_factor", type=int, default=5,
        help=(
            "When --rerank_template is active, recall top_k × rerank_factor "
            "candidates from the coarse search for template-matching re-rank "
            "(default: 5)."
        ),
    )

    # ── Local-mode arguments ───────────────────────────────────────────
    local = parser.add_argument_group("Local patch retrieval")
    local.add_argument(
        "--local_mode", action="store_true", default=False,
        help=(
            "Use dense multi-scale local-patch LBP matching instead of "
            "global-vector retrieval.  Each query patch votes independently "
            "and scores are aggregated by source image.  Requires "
            "build_index.py --local_index to have been run first."
        ),
    )
    local.add_argument(
        "--patch_size", type=int, default=DEFAULT_PATCH_SIZE,
        help=f"Square patch side length in pixels (default: {DEFAULT_PATCH_SIZE})"
    )
    local.add_argument(
        "--patch_stride", type=int, default=DEFAULT_PATCH_STRIDE,
        help=f"Stride between adjacent patches (default: {DEFAULT_PATCH_STRIDE})"
    )
    local.add_argument(
        "--patch_scales", type=str,
        default=",".join(str(s) for s in DEFAULT_PATCH_SCALES),
        help="Comma-separated scale factors for multi-scale sampling "
             "(default: 1.0,0.7,0.5)"
    )
    local.add_argument(
        "--top_per_patch", type=int, default=3,
        help="Top-N database patches considered per query patch during "
             "aggregation (default: 3)"
    )
    local.add_argument(
        "--auto_crop", action="store_true", default=False,
        help=(
            "Automatically detect and crop the clothing region from model "
            "photos before searching.  Uses face detection + body-proportion "
            "heuristics (no extra dependencies).  Recommended for model-shot "
            "→ fabric queries."
        ),
    )
    args = parser.parse_args()

    # ── Validate query path ───────────────────────────────────────────
    _query_path = Path(args.query)
    if not _query_path.exists():
        print(f"[ERROR] Query image not found: {args.query}")
        print(f"        Provide the full path, e.g.  py search.py images/0.jpg ...")
        sys.exit(1)

    # ── Load global index (always needed for metadata) ─────────────────
    try:
        global_features, global_meta = load_global_index()
    except FileNotFoundError:
        print("[ERROR] Global index not found. Run build_index.py first.")
        sys.exit(1)

    print(f"Index: {len(global_meta)} images, feature dim={global_features.shape[1]}")

    # ── Stage 1: coarse retrieval ──────────────────────────────────────
    if args.local_mode:
        # Local-patch retrieval
        try:
            coarse_results, n_query_patches = do_local_search(args, global_meta)
        except FileNotFoundError:
            print("[ERROR] Local index not found. Run: py build_index.py --local_index")
            sys.exit(1)
        mode_tag = f"local-{n_query_patches}p"
    else:
        # Global vector retrieval
        query_img = load_image(args.query)
        query_feat = extract_features(query_img)
        sims = cosine_similarity(query_feat, global_features)

        recall_k = (args.top_k * args.rerank_factor
                    if args.rerank_template else args.top_k)
        recall_k = min(recall_k, len(global_meta))

        order = np.argsort(-sims)
        coarse_results = []
        for i in order:
            score = float(sims[i])
            if score < args.threshold:
                break
            if len(coarse_results) >= recall_k:
                break
            coarse_results.append((int(i), score, global_meta[i]))
        mode_tag = "global"

    if not coarse_results:
        print("\n(no results above threshold)")
        return

    # ── Stage 2 (optional): template-matching rerank ───────────────────
    if args.rerank_template:
        rerank_scores = []

        print(f"\nTemplate-matching rerank: scanning {len(coarse_results)} "
              f"candidates ...")

        for idx, coarse_score, meta_item in coarse_results:
            candidate_path = str(IMAGES_DIR / meta_item["path"])
            try:
                tm_score, location = template_match_score(
                    args.query, candidate_path
                )
            except Exception as e:
                print(f"  [SKIP] {meta_item['path']} — {e}")
                continue
            rerank_scores.append(
                (idx, coarse_score, tm_score, location, meta_item)
            )

        if not rerank_scores:
            print("\n(no valid template matches)")
            return

        rerank_scores.sort(key=lambda x: x[2], reverse=True)
        results = rerank_scores[:args.top_k]

        # Terminal output
        print(f"\nTop-{args.top_k} results "
              f"({mode_tag} + template-matching rerank, "
              f"recall={len(coarse_results)}):")
        print("-" * 78)
        for rank, (idx, coarse_sim, tm_score, loc, m) in enumerate(results, 1):
            print(
                f"  #{rank}  coarse={coarse_sim:.6f}  tm={tm_score:.4f}  "
                f"loc=({loc[0]},{loc[1]})  |  {m['path']}"
            )

        title = (
            f"Texture Search  [{mode_tag} + template-matching rerank]\n"
            f"Query: {Path(args.query).name}"
        )
        visualize_rerank(args.query, results, title, args.output)

    else:
        # ── Single-stage output ────────────────────────────────────────
        results = coarse_results[:args.top_k]

        print(f"\nTop-{args.top_k} results ({mode_tag}, "
              f"threshold={args.threshold}):")
        print("-" * 60)
        for rank, (idx, score, m) in enumerate(results, 1):
            print(f"  #{rank}  sim={score:.6f}  |  {m['path']}")

        if results:
            title = (
                f"Texture Similarity Search  [{mode_tag}]\n"
                f"Query: {Path(args.query).name}"
            )
            visualize_default(args.query, results, title, args.output)


if __name__ == "__main__":
    main()
