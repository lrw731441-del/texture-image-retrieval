"""
Enhanced feature extractor — aspect-ratio-preserving resize, multi-radius LBP,
spatial-pyramid HOG, DCT perceptual hash (pHash), dual-layer CNN with
multi-scale grid pooling. Group-wise L2 normalization + weighted fusion.
All I/O handles Unicode paths on Windows.

Also provides dense multi-scale local patch extraction and lightweight
texture descriptors for patch-level retrieval.
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Compose, ToTensor, Normalize

# ── Constants ──────────────────────────────────────────────────────────────

TARGET_MIN_SIDE = 128

# LBP — rotation-invariant uniform, 3 radii
LBP_RADII = [1, 2, 3]
LBP_METHOD = "uniform"

# HOG — coarser cells → compact, spatially robust
HOG_WIN_SIZE = (128, 128)
HOG_CELL_SIZE = (16, 16)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (16, 16)
HOG_N_BINS = 9                              # → 64 × 9 = 576 dims per region

# pHash — DCT low-frequency coefficients
PHASH_DCT_SIZE = 8                          # top-left 8×8, excluding DC → 63 dims

# Image pyramid scales for LBP/HOG
PYRAMID_SCALES = [1.0, 1.0 / np.sqrt(2), 0.5]

# Group weights: LBP dominates (texture is the primary matching signal),
# HOG and pHash provide supporting structure, CNN is deprioritised
# because ImageNet semantics often disagree with texture similarity.
GROUP_WEIGHTS = [4.0, 1.5, 1.5, 0.5]       # LBP, HOG, pHash, CNN

# Local patch defaults
DEFAULT_PATCH_SIZE = 48
DEFAULT_PATCH_STRIDE = 24
DEFAULT_PATCH_SCALES = [1.0, 0.7, 0.5]
LOCAL_LBP_RADII = [1, 2, 3]                 # → 10 + 18 + 26 = 54 dims


# ── I/O ────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load BGR image with Unicode path support.  Falls back to PIL when
    OpenCV cannot decode the format (e.g. HEIC, AVIF, some WebP variants)."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image
            pil_img = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(
                f"Cannot read image (unsupported format or corrupt): {path}"
            )
    return img


# ── Clothing-region extraction (model-photo → fabric crop) ────────────────

def extract_clothing_region(img: np.ndarray) -> np.ndarray:
    """
    Automatically crop the clothing region from a model photo.

    Uses a generous centre-torso crop (~60% height × ~84% width).
    When face detection succeeds the vertical position is refined so the
    crop is centred on the torso (~3 head-heights below the chin), but
    the crop *size* stays large enough to cover the full garment.
    Falls back to the centre crop when face detection is unavailable.
    Does NOT require any extra dependencies.
    """
    h, w = img.shape[:2]

    # ── Default: generous centre-torso crop ─────────────────────────
    crop_h = int(h * 0.60)
    crop_w = int(w * 0.84)
    cy = h // 2                     # centre vertically
    cx = w // 2

    # ── Face-guided vertical refinement ─────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier()

    if face_cascade.load(cascade_path):
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(faces) > 0:
            _, fy, _, fh = max(faces, key=lambda r: r[2] * r[3])
            # Torso centre ≈ 3 head-heights below chin
            cy = fy + fh + int(fh * 3.0)
            # Ensure crop stays within image bounds
            if crop_h > h:
                crop_h = h

    # Clip to image
    top = max(0, cy - crop_h // 2)
    bottom = min(h, top + crop_h)
    left = max(0, cx - crop_w // 2)
    right = min(w, left + crop_w)

    return img[top:bottom, left:right]


# ── Template matching ──────────────────────────────────────────────────────

def template_match_score(query_path: str, candidate_path: str):
    """
    Normalised cross-correlation template matching (TM_CCOEFF_NORMED).
    Handles Unicode paths via np.fromfile + cv2.imdecode.

    If the query is larger than the candidate in either dimension, the query
    is automatically scaled to 80% of the candidate's size (aspect-ratio
    preserved) so the template always fits inside the candidate.

    Returns (max_score, (x, y)):
      - max_score  — best correlation in [-1, 1], higher is better
      - (x, y)     — top-left corner of the match in the candidate image
    """
    with open(query_path, "rb") as f:
        q_data = np.frombuffer(f.read(), dtype=np.uint8)
    query_gray = cv2.imdecode(q_data, cv2.IMREAD_GRAYSCALE)
    if query_gray is None:
        try:
            from PIL import Image
            query_gray = np.array(Image.open(query_path).convert("L"))
        except Exception:
            raise ValueError(f"Cannot read query image: {query_path}")

    with open(candidate_path, "rb") as f:
        c_data = np.frombuffer(f.read(), dtype=np.uint8)
    cand_gray = cv2.imdecode(c_data, cv2.IMREAD_GRAYSCALE)
    if cand_gray is None:
        try:
            from PIL import Image
            cand_gray = np.array(Image.open(candidate_path).convert("L"))
        except Exception:
            raise ValueError(f"Cannot read candidate image: {candidate_path}")

    q_h, q_w = query_gray.shape
    c_h, c_w = cand_gray.shape

    if q_h > c_h or q_w > c_w:
        scale = 0.8 * min(c_h / q_h, c_w / q_w)
        new_h = max(1, int(round(q_h * scale)))
        new_w = max(1, int(round(q_w * scale)))
        query_gray = cv2.resize(query_gray, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)

    result = cv2.matchTemplate(cand_gray, query_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return (float(max_val), max_loc)


# ── Fabric augmentation (simulates wearing for cross-domain matching) ────

def augment_fabric(img: np.ndarray, n_variants: int = 4,
                   seed: int = 0) -> list:
    """
    Generate simulated 'worn' versions of a flat fabric scan using
    perspective warp + elastic deformation + gamma correction.

    Each variant helps bridge the domain gap between flat 2-D scans
    and fabric photographed on a body (draped, lit, angled).

    Args:
        img:         BGR image (any size).
        n_variants:  Number of augmented copies to return.
        seed:        Fixed seed per image for reproducibility.

    Returns:
        List of BGR uint8 arrays, each the same shape as *img*.
    """
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    variants = []

    for _ in range(n_variants):
        aug = img.copy()

        # 1. Subtle perspective warp — simulates off-angle camera
        margin = 0.10
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dst = np.float32([
            [w * rng.uniform(0, margin), h * rng.uniform(0, margin)],
            [w * rng.uniform(1 - margin, 1), h * rng.uniform(0, margin)],
            [w * rng.uniform(0, margin), h * rng.uniform(1 - margin, 1)],
            [w * rng.uniform(1 - margin, 1), h * rng.uniform(1 - margin, 1)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        aug = cv2.warpPerspective(aug, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT)

        # 2. Elastic deformation — simulates fabric draping / folds
        alpha = w * 0.04
        sigma = w * 0.04
        dx = cv2.GaussianBlur(
            rng.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma)
        dy = cv2.GaussianBlur(
            rng.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma)
        map_x = np.arange(w, dtype=np.float32).reshape(1, -1) + dx
        map_y = np.arange(h, dtype=np.float32).reshape(-1, 1) + dy
        aug = cv2.remap(aug, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

        # 3. Gamma correction — simulates lighting variation
        gamma = rng.uniform(0.7, 1.5)
        lut = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
        aug = cv2.LUT(aug, lut)

        variants.append(aug)

    return variants


# ── Dense local patch extraction ───────────────────────────────────────────

def extract_dense_patches(gray: np.ndarray,
                          scales: list = None,
                          patch_size: int = DEFAULT_PATCH_SIZE,
                          stride: int = DEFAULT_PATCH_STRIDE):
    """
    Multi-scale dense grid patch sampling from a grayscale image.

    Args:
        gray:       Grayscale image (H×W).
        scales:     Scale factors applied to the image before sampling.
        patch_size: Side length of each square patch in pixels.
        stride:     Step between adjacent patch top-left corners.

    Returns:
        List of (patch_array, (x, y, w, h), scale) where patch_array is a
        2-D uint8 array of shape (h, w).
    """
    if scales is None:
        scales = DEFAULT_PATCH_SCALES

    patches = []
    h, w = gray.shape

    for scale in scales:
        scaled_h = max(1, int(round(h * scale)))
        scaled_w = max(1, int(round(w * scale)))
        scaled_gray = cv2.resize(gray, (scaled_w, scaled_h),
                                 interpolation=cv2.INTER_AREA)

        eff_ps = min(patch_size, scaled_h, scaled_w)
        eff_stride = max(1, min(stride, int(round(stride * scale))))

        for y in range(0, scaled_h - eff_ps + 1, eff_stride):
            for x in range(0, scaled_w - eff_ps + 1, eff_stride):
                patch = scaled_gray[y:y + eff_ps, x:x + eff_ps]
                patches.append((patch, (x, y, eff_ps, eff_ps), scale))

    return patches


def extract_local_texture(patch_gray: np.ndarray) -> np.ndarray:
    """
    Lightweight local texture descriptor — multi-radius LBP (R=1,2,3,
    rotation-invariant uniform).  No CNN, no HOG — deliberately compact
    so a single image can be represented by hundreds of patch vectors
    without blowing up the index.

    Returns a 54-dim L2-normalised float32 vector.
    """
    feats = []
    for r in LOCAL_LBP_RADII:
        n_points = 8 * r
        lbp = local_binary_pattern(patch_gray, P=n_points, R=r,
                                   method=LBP_METHOD)
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)
        feats.append(hist)
    feat = np.concatenate(feats).astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


# ── Preprocessing ──────────────────────────────────────────────────────────

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Direct resize to 128×128. Deliberately ignores aspect ratio to preserve
    ALL image content — slight distortion is acceptable because LBP+HOG
    capture local texture patterns independent of global shape.
    """
    return cv2.resize(img, (TARGET_MIN_SIDE, TARGET_MIN_SIDE),
                       interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ── LBP: multi-radius × image pyramid ──────────────────────────────────────

def _lbp_bins(radius: int) -> int:
    return 8 * radius + 2


def extract_lbp(gray: np.ndarray, radius: int) -> np.ndarray:
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, P=n_points, R=radius, method=LBP_METHOD)
    n_bins = _lbp_bins(radius)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_lbp_multiscale(gray: np.ndarray) -> np.ndarray:
    """3 radii × 3 image scales = 9 LBP histograms."""
    feats = []
    base = TARGET_MIN_SIDE
    for r in LBP_RADII:
        for scale in PYRAMID_SCALES:
            s = max(int(round(base * scale)), 8 * r)
            scaled = cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA)
            feats.append(extract_lbp(scaled, r))
    return np.concatenate(feats).astype(np.float32)


# ── HOG: spatial pyramid × image pyramid ───────────────────────────────────

_hog = cv2.HOGDescriptor(HOG_WIN_SIZE, HOG_BLOCK_SIZE, HOG_BLOCK_STRIDE,
                          HOG_CELL_SIZE, HOG_N_BINS)


def extract_hog_region(gray: np.ndarray) -> np.ndarray:
    """HOG on a single region (resized to WIN_SIZE). L2-normalized."""
    if gray.shape != HOG_WIN_SIZE:
        gray = cv2.resize(gray, HOG_WIN_SIZE, interpolation=cv2.INTER_AREA)
    desc = _hog.compute(gray)
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc = desc / norm
    return desc.ravel().astype(np.float32)


def extract_hog_spatial_pyramid(gray: np.ndarray) -> np.ndarray:
    """1×1 (global) + 2×2 (quadrants) HOG descriptors."""
    feats = [extract_hog_region(gray)]
    h, w = gray.shape
    for r in range(2):
        for c in range(2):
            y0, y1 = r * h // 2, (r + 1) * h // 2
            x0, x1 = c * w // 2, (c + 1) * w // 2
            feats.append(extract_hog_region(gray[y0:y1, x0:x1]))
    return np.concatenate(feats).astype(np.float32)


def extract_hog_multiscale(gray: np.ndarray) -> np.ndarray:
    """Spatial-pyramid HOG at 3 image scales."""
    feats = []
    base = TARGET_MIN_SIDE
    for scale in PYRAMID_SCALES:
        s = max(int(round(base * scale)), 32)
        scaled = cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA)
        feats.append(extract_hog_spatial_pyramid(scaled))
    return np.concatenate(feats).astype(np.float32)


# ── pHash: DCT low-frequency coefficients ──────────────────────────────────

def extract_phash(gray: np.ndarray) -> np.ndarray:
    """
    DCT perceptual hash — takes |DCT| low-frequency coefficients from a 32×32
    thumbnail.  Absolute values avoid negative dot-products in cosine space
    and capture coarse image structure robust to JPEG compression and shifts.
    Returns 63-dim non-negative float vector.
    """
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(thumb))
    coeffs = np.abs(dct[:PHASH_DCT_SIZE, :PHASH_DCT_SIZE]).ravel()
    feat = coeffs[1:]                                          # 63 dims, skip DC
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat.astype(np.float32)


# ── CNN: dual-layer + multi-scale grid pooling ─────────────────────────────

class MultiScaleGridPool(nn.Module):
    """Divides feature map into 1×1, 2×2, 4×4 grids, averages each cell, concatenates."""

    def __init__(self, in_channels: int, include_4x4: bool = True):
        super().__init__()
        self.p1 = nn.AdaptiveAvgPool2d((1, 1))
        self.p2 = nn.AdaptiveAvgPool2d((2, 2))
        self.include_4x4 = include_4x4
        if include_4x4:
            self.p4 = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [self.p1(x).flatten(1), self.p2(x).flatten(1)]
        if self.include_4x4:
            parts.append(self.p4(x).flatten(1))
        return torch.cat(parts, dim=1)


_cnn_models = None
_cnn_device = None


def _get_cnn_models():
    global _cnn_models, _cnn_device
    if _cnn_models is None:
        _cnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        children = list(backbone.children())

        # layer3: 256 ch, 8×8 map for 128×128 input
        trunk3 = nn.Sequential(*children[:-3]).eval().to(_cnn_device)
        model3 = nn.Sequential(
            trunk3,
            MultiScaleGridPool(256, include_4x4=True),   # (1+4+16)×256 = 5376
        ).eval().to(_cnn_device)

        # layer4: 512 ch, 4×4 map for 128×128 input
        trunk4 = nn.Sequential(*children[:-2]).eval().to(_cnn_device)
        model4 = nn.Sequential(
            trunk4,
            MultiScaleGridPool(512, include_4x4=True),   # (1+4+16)×512 = 10752
        ).eval().to(_cnn_device)

        for m in [model3, model4]:
            for p in m.parameters():
                p.requires_grad = False

        _cnn_models = (model3, model4)
    return _cnn_models, _cnn_device


_cnn_tf = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract_cnn_features(rgb: np.ndarray) -> np.ndarray:
    """
    Dual-layer CNN: layer3 (5376 dims) + layer4 (10752 dims) = 16128 dims.
    """
    (model3, model4), device = _get_cnn_models()
    tensor = _cnn_tf(rgb).unsqueeze(0).to(device)
    f3 = model3(tensor)
    f4 = model4(tensor)
    return torch.cat([f3, f4], dim=1).squeeze(0).cpu().numpy().astype(np.float32)


# ── Main entry ─────────────────────────────────────────────────────────────

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Full pipeline:
    1. Direct resize → 128×128
    2. Multi-radius LBP (R=1,2,3) × 3 image scales      → L2 → × w[0]
    3. Spatial-pyramid HOG (1×1+2×2) × 3 image scales   → L2 → × w[1]
    4. DCT pHash (8×8 low-freq coeffs, no DC)           → L2 → × w[2]
    5. CNN layer3+4, each with 1×1+2×2+4×4 grid pool    → L2 → × w[3]
    6. Weighted concatenation → final L2 normalization

    Weights favour texture+structure over semantics:
      LBP ×4, HOG ×1.5, pHash ×1.5, CNN ×0.5
    """
    processed = preprocess_image(img)
    gray = to_gray(processed)
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    groups = [
        extract_lbp_multiscale(gray),
        extract_hog_multiscale(gray),
        extract_phash(gray),
        extract_cnn_features(rgb),
    ]

    weighted = []
    for w, g in zip(GROUP_WEIGHTS, groups):
        weighted.append(w * _l2_norm(g))

    return _l2_norm(np.concatenate(weighted))
