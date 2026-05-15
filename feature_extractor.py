"""
LBP + HOG mixed scale-invariant texture feature extractor.
Resizes all images to 128x128, converts to grayscale,
extracts rotation-invariant uniform LBP + HOG, and concatenates.
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern


TARGET_SIZE = 128

# LBP params — rotation-invariant uniform pattern
LBP_RADIUS = 2
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

# HOG params
HOG_WIN_SIZE = (128, 128)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_N_BINS = 9


def _resize_and_gray(img: np.ndarray) -> np.ndarray:
    """Resize to TARGET_SIZE x TARGET_SIZE and convert to grayscale."""
    resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    if resized.ndim == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    elif resized.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    return resized


def extract_lbp(gray: np.ndarray) -> np.ndarray:
    """Extract rotation-invariant uniform LBP histogram."""
    lbp = local_binary_pattern(
        gray, P=LBP_N_POINTS, R=LBP_RADIUS, method=LBP_METHOD
    )
    n_bins = LBP_N_POINTS + 2  # "uniform" bins count
    hist, _ = np.histogram(
        lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
    )
    return hist.astype(np.float32)


def extract_hog(gray: np.ndarray) -> np.ndarray:
    """Extract HOG descriptor."""
    hog = cv2.HOGDescriptor(
        _winSize=HOG_WIN_SIZE,
        _blockSize=HOG_BLOCK_SIZE,
        _blockStride=HOG_BLOCK_STRIDE,
        _cellSize=HOG_CELL_SIZE,
        _nbins=HOG_N_BINS,
    )
    desc = hog.compute(gray)
    # L2-normalize for scale invariance
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc = desc / norm
    return desc.ravel().astype(np.float32)


def extract_features(img: np.ndarray) -> np.ndarray:
    """Extract concatenated LBP+HOG feature vector."""
    gray = _resize_and_gray(img)
    feat_lbp = extract_lbp(gray)
    feat_hog = extract_hog(gray)
    return np.concatenate([feat_lbp, feat_hog])


def load_image(path: str) -> np.ndarray:
    """Load image with Unicode path support."""
    # cv2.imread fails on some unicode paths on Windows — use imdecode
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img
