import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


Color = Tuple[int, int, int]

STYLE_PALETTES: Dict[str, List[Color]] = {
    "classic": [(0, 0, 0), (255, 255, 255), (128, 128, 128), (192, 192, 192), (128, 0, 0), (255, 0, 0), (128, 128, 0), (255, 255, 0), (0, 128, 0), (0, 255, 0), (0, 128, 128), (0, 255, 255), (0, 0, 128), (0, 0, 255), (128, 0, 128), (255, 0, 255)],
    "hd": [(15, 15, 20), (38, 43, 68), (58, 68, 102), (88, 111, 135), (126, 159, 157), (170, 205, 170), (224, 248, 207), (255, 255, 255), (255, 204, 170), (255, 170, 170), (238, 118, 95), (204, 68, 75), (153, 52, 65), (89, 38, 73), (50, 25, 50), (20, 12, 28)],
    "gameboy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "nes": [(124, 124, 124), (0, 0, 252), (0, 0, 188), (68, 40, 188), (148, 0, 132), (168, 0, 32), (168, 16, 0), (136, 20, 0), (80, 48, 0), (0, 120, 0), (0, 104, 0), (0, 88, 0), (0, 64, 88), (0, 0, 0), (188, 188, 188), (248, 248, 248)],
    "snes": [(33, 30, 32), (94, 36, 88), (155, 50, 89), (201, 87, 60), (238, 166, 69), (246, 213, 92), (237, 238, 158), (168, 219, 168), (79, 193, 129), (40, 143, 119), (45, 83, 130), (81, 60, 119), (145, 102, 172), (201, 150, 199), (235, 206, 213), (255, 255, 255)],
    "minecraft": [(89, 125, 39), (109, 153, 48), (127, 178, 56), (151, 109, 77), (112, 78, 55), (90, 63, 45), (134, 96, 67), (96, 96, 96), (128, 128, 128), (160, 160, 160), (216, 175, 147), (180, 136, 107), (71, 45, 60), (57, 41, 35), (30, 27, 26), (237, 237, 237)],
    "lego": [(242, 205, 55), (201, 26, 9), (0, 85, 191), (35, 120, 65), (88, 42, 18), (27, 42, 52), (255, 255, 255), (161, 165, 162), (109, 110, 108), (0, 0, 0), (254, 138, 24), (180, 210, 227), (51, 0, 114), (255, 158, 205), (187, 233, 11), (129, 0, 123)],
    "pico8": [(0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81), (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232), (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54), (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)],
    "c64": [(0, 0, 0), (255, 255, 255), (136, 0, 0), (170, 255, 238), (204, 68, 204), (0, 204, 85), (0, 0, 170), (238, 238, 119), (221, 136, 85), (102, 68, 0), (255, 119, 119), (51, 51, 51), (119, 119, 119), (170, 255, 102), (0, 136, 255), (187, 187, 187)],
}

STYLE_ALIASES = {"pixel": "classic", "hd": "hd", "gameboy": "gameboy", "nes": "nes", "snes": "snes", "minecraft": "minecraft", "lego": "lego", "pico8": "pico8", "c64": "c64", "commodore64": "c64"}
STYLE_NAMES = {"classic": "Classic", "hd": "HD Pixel", "gameboy": "GameBoy", "nes": "NES", "snes": "SNES", "minecraft": "Minecraft", "lego": "LEGO", "pico8": "Pico-8", "c64": "Commodore64"}

OUTLINE_LEVELS = {"off": 0, "thin": 1, "medium": 2, "thick": 3}

@dataclass
class ProcessingOptions:
    style: str = "classic"
    pixel_size: int = 16
    palette_size: int = 16
    outline: str = "off"
    dithering: bool = False
    denoise: bool = False
    gamma: float = 1.0


def _palette_array(style: str, palette_size: int) -> np.ndarray:
    palette = STYLE_PALETTES.get(style, STYLE_PALETTES["classic"])
    if palette_size <= len(palette):
        colors = palette[:palette_size]
    else:
        reps = int(np.ceil(palette_size / len(palette)))
        colors = (palette * reps)[:palette_size]
    return np.asarray(colors, dtype=np.float32)


def apply_auto_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 0.01)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def quantize_to_palette(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    flat = image.reshape((-1, 3)).astype(np.float32)
    distances = np.sum((flat[:, None, :] - palette[None, :, :]) ** 2, axis=2)
    return palette[np.argmin(distances, axis=1)].reshape(image.shape).astype(np.uint8)


def kmeans_color_quantization(image: np.ndarray, clusters: int) -> np.ndarray:
    pixels = image.reshape((-1, 3)).astype(np.float32)
    k = max(1, min(int(clusters), len(pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    return centers[labels.flatten()].reshape(image.shape).astype(np.uint8)


def floyd_steinberg(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    work = image.astype(np.float32).copy()
    h, w = work.shape[:2]
    for y in range(h):
        for x in range(w):
            old = work[y, x].copy()
            new = palette[np.argmin(np.sum((palette - old) ** 2, axis=1))]
            work[y, x] = new
            err = old - new
            if x + 1 < w: work[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x > 0: work[y + 1, x - 1] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if x + 1 < w: work[y + 1, x + 1] += err * 1 / 16
    return np.clip(work, 0, 255).astype(np.uint8)


def add_outline(image: np.ndarray, thickness: int) -> np.ndarray:
    if thickness <= 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    kernel = np.ones((thickness, thickness), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    result = image.copy()
    result[edges > 0] = (0, 0, 0)
    return result


def process_image(input_file: str, output_file: str, options: ProcessingOptions) -> str:
    bgr = cv2.imread(input_file, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to read image file")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    max_side = 1600
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
    image = apply_auto_contrast(image)
    image = apply_clahe(image)
    image = sharpen(image)
    if options.denoise:
        image = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), None, 5, 5, 7, 21)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = gamma_correct(image, options.gamma)
    small_w = max(1, w // options.pixel_size)
    small_h = max(1, h // options.pixel_size)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small = kmeans_color_quantization(small, options.palette_size)
    palette = _palette_array(options.style, options.palette_size)
    small = floyd_steinberg(small, palette) if options.dithering else quantize_to_palette(small, palette)
    small = add_outline(small, OUTLINE_LEVELS.get(options.outline, 0))
    result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(result).save(output_file)
    return output_file


def pixel_art(input_file, output_file, pixel_size=16):
    return process_image(input_file, output_file, ProcessingOptions(pixel_size=pixel_size))


def create_comparison(input_file: str, processed_file: str, output_file: str) -> str:
    before = Image.open(input_file).convert("RGB")
    after = Image.open(processed_file).convert("RGB")
    before.thumbnail((700, 700), Image.Resampling.LANCZOS)
    after.thumbnail((700, 700), Image.Resampling.LANCZOS)
    h = max(before.height, after.height) + 50
    w = before.width + after.width + 20
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Before", fill="black")
    draw.text((before.width + 20, 10), "After", fill="black")
    canvas.paste(before, (0, 50))
    canvas.paste(after, (before.width + 20, 50))
    canvas.save(output_file)
    return output_file


def create_palette_preview(style: str, palette_size: int, output_file: str) -> str:
    colors = _palette_array(style, palette_size).astype(np.uint8)
    swatch = 48
    cols = min(8, len(colors))
    rows = int(np.ceil(len(colors) / cols))
    img = Image.new("RGB", (cols * swatch, rows * swatch), "white")
    draw = ImageDraw.Draw(img)
    for i, color in enumerate(colors):
        x, y = (i % cols) * swatch, (i // cols) * swatch
        draw.rectangle([x, y, x + swatch, y + swatch], fill=tuple(map(int, color)))
    img.save(output_file)
    return output_file


def get_image_info(path: str) -> Dict[str, str]:
    with Image.open(path) as img:
        stat = os.stat(path)
        return {"format": img.format or "Unknown", "size": f"{img.width}×{img.height}", "mode": img.mode, "file_size": f"{stat.st_size / 1024:.1f} KB"}


def cleanup_temp(paths: Iterable[str], older_than_seconds: int = 3600) -> None:
    now = time.time()
    for folder in paths:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and now - os.path.getmtime(path) > older_than_seconds:
                try:
                    os.remove(path)
                except OSError:
                    pass
