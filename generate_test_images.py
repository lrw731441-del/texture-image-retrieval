"""Generate synthetic test images — same texture in different colors."""

import cv2
import numpy as np
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent / "images"
SIZE = 256  # will be resized to 128 by the pipeline


def _save(name, img):
    path = IMAGES_DIR / name
    cv2.imwrite(str(path), img)
    print(f"  {name}")


def make_stripe(angle_deg, color1, color2):
    """Rotated stripe pattern."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for i in range(SIZE):
        for j in range(SIZE):
            # Rotate coordinate
            rad = np.deg2rad(angle_deg)
            xr = i * np.cos(rad) - j * np.sin(rad)
            val = 255 if int(xr) % 40 < 20 else 128
            img[i, j] = color1 if int(xr) % 40 < 20 else color2
    return img


def make_grid(spacing, color1, color2):
    """Grid / checker-like pattern."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for i in range(SIZE):
        for j in range(SIZE):
            val = color1 if (i // spacing + j // spacing) % 2 == 0 else color2
            img[i, j] = val
    return img


def make_circles(n, r, color1, color2):
    """Random circle texture."""
    img = np.full((SIZE, SIZE, 3), color2, dtype=np.uint8)
    rng = np.random.RandomState(42)
    for _ in range(n):
        cx, cy = rng.randint(0, SIZE), rng.randint(0, SIZE)
        cv2.circle(img, (cx, cy), r, (int(color1[0]), int(color1[1]), int(color1[2])), -1)
    return img


def make_perlin_like(color1, color2):
    """Blob-like texture using blurred random dots."""
    rng = np.random.RandomState(7)
    img = rng.randint(0, 256, (SIZE // 4, SIZE // 4)).astype(np.float32)
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (21, 21), 10)
    img = (img / img.max() * 255).astype(np.uint8)
    out = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    out[:, :, 0] = (img / 255.0 * color1[0]).astype(np.uint8)
    out[:, :, 1] = (img / 255.0 * color1[1]).astype(np.uint8)
    out[:, :, 2] = (img / 255.0 * color1[2]).astype(np.uint8)
    return out


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating test images ...")

    # Group 1: stripes — same texture, different colors
    _save("stripe_red_cyan.png", make_stripe(30, (0, 0, 255), (255, 255, 0)))
    _save("stripe_blue_green.png", make_stripe(30, (255, 0, 0), (0, 255, 0)))
    _save("stripe_purple_yellow.png", make_stripe(30, (255, 0, 255), (0, 255, 255)))

    # Group 2: grid — different texture family
    _save("grid_black_white.png", make_grid(32, (0, 0, 0), (255, 255, 255)))
    _save("grid_red_blue.png", make_grid(32, (0, 0, 255), (255, 0, 0)))

    # Group 3: circles
    _save("circles_warm.png", make_circles(40, 18, (0, 100, 255), (255, 200, 100)))
    _save("circles_cool.png", make_circles(40, 18, (255, 100, 0), (100, 200, 255)))

    # Group 4: blob texture
    _save("blob_orange.png", make_perlin_like((0, 140, 255), (0, 0, 0)))
    _save("blob_green.png", make_perlin_like((0, 255, 140), (0, 0, 0)))
    _save("blob_purple.png", make_perlin_like((255, 0, 140), (0, 0, 0)))

    print("Done. 10 test images created.")


if __name__ == "__main__":
    main()
