"""Synthetic smoke tests for the Pixel Art Telegram bot image pipeline.

The script creates representative inputs and writes original, processed, and comparison
PNG files under test_outputs/. It also verifies that every processed output is a real
nearest-neighbor pixel grid with a bounded palette instead of a blurred resize.
"""
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from image_tools import ProcessingOptions, create_comparison, process_image

import cv2
import numpy as np
from PIL import Image, ImageDraw

from image_tools import ProcessingOptions, create_comparison, process_image

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "test_outputs" / "originals"
OUTPUT_DIR = ROOT / "test_outputs" / "processed"
COMPARE_DIR = ROOT / "test_outputs" / "comparisons"


def make_gradient(size, dark=False, bright=False, alpha=False):
    w, h = size
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    img = np.dstack([xx, yy, ((xx.astype(int) + yy.astype(int)) // 2).astype(np.uint8)])
    if dark:
        img = (img * 0.28).astype(np.uint8)
    if bright:
        img = np.clip(180 + img * 0.30, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, "RGB")
    draw = ImageDraw.Draw(pil)
    draw.rectangle([w // 8, h // 8, w // 2, h // 2], outline="white", width=max(1, min(w, h) // 40))
    draw.ellipse([w // 2, h // 3, w - w // 8, h - h // 8], fill=(220, 60, 80), outline="black", width=max(1, min(w, h) // 50))
    if not alpha:
        return pil
    rgba = pil.convert("RGBA")
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse([w // 5, h // 5, w - w // 5, h - h // 5], fill=255)
    rgba.putalpha(mask)
    return rgba


def assert_pixel_grid(path: Path, max_colors: int):
    arr = np.asarray(Image.open(path).convert("RGBA"))
    rgb = arr[:, :, :3]
    unique_colors = np.unique(rgb.reshape(-1, 3), axis=0)
    assert len(unique_colors) <= max_colors + 4, f"{path.name} has too many colors: {len(unique_colors)}"
    # Real pixel art has many identical neighboring pixels after nearest-neighbor upscaling.
    same_x = np.mean(np.all(rgb[:, 1:] == rgb[:, :-1], axis=2))
    same_y = np.mean(np.all(rgb[1:, :] == rgb[:-1, :], axis=2))
    assert max(same_x, same_y) > 0.80, f"{path.name} does not look blocky enough"


def main():
    for folder in (INPUT_DIR, OUTPUT_DIR, COMPARE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    cases = {
        "portrait": make_gradient((320, 640)),
        "landscape": make_gradient((800, 420)),
        "low_resolution": make_gradient((96, 72)),
        "high_resolution": make_gradient((1800, 1200)),
        "transparent_png": make_gradient((360, 360), alpha=True),
        "dark_image": make_gradient((420, 420), dark=True),
        "bright_image": make_gradient((420, 420), bright=True),
    }
    options = ProcessingOptions(style="hd", quality="hd", pixel_size=None, palette_size=16, outline="thin", dithering=True, sharpen=True)

    for name, image in cases.items():
        input_path = INPUT_DIR / f"{name}.png"
        output_path = OUTPUT_DIR / f"{name}_pixel.png"
        compare_path = COMPARE_DIR / f"{name}_comparison.png"
        image.save(input_path)
        process_image(str(input_path), str(output_path), options)
        create_comparison(str(input_path), str(output_path), str(compare_path))
        assert output_path.exists(), output_path
        assert compare_path.exists(), compare_path
        assert_pixel_grid(output_path, options.palette_size)

    print(f"Generated {len(cases)} originals, processed images, and comparisons in {ROOT / 'test_outputs'}")


if __name__ == "__main__":
    main()
