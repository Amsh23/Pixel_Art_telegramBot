import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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
    "retro": [(8, 8, 16), (32, 24, 48), (64, 40, 88), (96, 56, 112), (144, 72, 120), (192, 88, 112), (224, 128, 96), (248, 184, 120), (255, 224, 168), (216, 248, 184), (144, 216, 144), (80, 176, 136), (56, 120, 152), (72, 80, 160), (128, 112, 184), (224, 192, 216)],
}

STYLE_ALIASES = {"pixel": "classic", "hd": "hd", "hd_pixel": "hd", "retro": "retro", "gameboy": "gameboy", "nes": "nes", "snes": "snes", "minecraft": "minecraft", "lego": "lego", "pico8": "pico8", "c64": "c64", "commodore64": "c64"}
STYLE_NAMES = {"classic": "Classic", "hd": "HD Pixel", "gameboy": "GameBoy", "nes": "NES", "snes": "SNES", "minecraft": "Minecraft", "lego": "LEGO", "pico8": "Pico-8", "c64": "Commodore64", "retro": "Retro RPG"}
QUALITY_MODES = ("fast", "normal", "hd")
GENERATION_MODES = ("classic", "ai")
OUTLINE_LEVELS = {"off": 0, "thin": 1, "medium": 2, "thick": 3}

@dataclass
class ImageAnalysis:
    width: int
    height: int
    aspect_ratio: float
    complexity: float
    brightness: float
    chosen_pixel_size: int
    grid_size: Tuple[int, int]
    palette_size: int
    dominant_colors: List[Color]

@dataclass
class ProcessingOptions:
    style: str = "classic"
    pixel_size: Optional[int] = None  # None means auto-detect from resolution and detail complexity.
    palette_size: int = 16
    quality: str = "normal"
    outline: str = "off"
    dithering: bool = True
    sharpen: bool = True
    denoise: bool = False
    gamma: float = 1.0
    generation_mode: str = "classic"
    ai_prompt: str = ""
    ai_strength: float = 0.55
    ai_steps: int = 28


def _palette_array(style: str, palette_size: int) -> np.ndarray:
    palette = STYLE_PALETTES.get(style, STYLE_PALETTES["classic"])
    if palette_size <= len(palette):
        colors = palette[:palette_size]
    else:
        reps = int(np.ceil(palette_size / len(palette)))
        colors = (palette * reps)[:palette_size]
    return np.asarray(colors, dtype=np.float32)


def _read_image(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with Image.open(path) as pil:
        rgba = pil.convert("RGBA")
    arr = np.asarray(rgba)
    return arr[:, :, :3].copy(), arr[:, :, 3].copy()


def dominant_colors(image: np.ndarray, count: int = 8) -> List[Color]:
    sample = cv2.resize(image, (min(96, image.shape[1]), min(96, image.shape[0])), interpolation=cv2.INTER_AREA)
    q = kmeans_palette(sample, count)
    return [tuple(map(int, c)) for c in q]


def calculate_complexity(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0
    return float(np.clip(edge_density * 1.8 + texture, 0.0, 1.0))


def choose_pixel_size(width: int, height: int, complexity: float) -> int:
    longest = max(width, height)
    if longest <= 256:
        base = 4
    elif longest <= 700:
        base = 8
    elif longest <= 1400:
        base = 16
    else:
        base = 32
    if complexity > 0.45 and longest > 700:
        base *= 2
    elif complexity < 0.12 and longest <= 1000:
        base = max(4, base // 2)
    return int(np.clip(base, 4, 64))


def analyze_image_array(image: np.ndarray, requested_pixel_size: Optional[int], palette_size: int) -> ImageAnalysis:
    h, w = image.shape[:2]
    complexity = calculate_complexity(image)
    block = requested_pixel_size or choose_pixel_size(w, h, complexity)
    grid = (max(1, w // block), max(1, h // block))
    return ImageAnalysis(w, h, w / max(h, 1), complexity, float(np.mean(image)), block, grid, palette_size, dominant_colors(image, min(8, palette_size)))


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
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    return cv2.addWeighted(image, 1.55, blurred, -0.55, 0)


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 0.01)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def kmeans_palette(image: np.ndarray, clusters: int) -> np.ndarray:
    pixels = image.reshape((-1, 3)).astype(np.float32)
    if len(pixels) > 25000:
        pixels = pixels[np.random.default_rng(42).choice(len(pixels), 25000, replace=False)]
    k = max(1, min(int(clusters), len(pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return np.clip(centers, 0, 255).astype(np.float32)


def kmeans_color_quantization(image: np.ndarray, clusters: int) -> np.ndarray:
    """Backward-compatible helper that quantizes an image to a learned K-Means palette."""
    palette = kmeans_palette(image, clusters)
    return quantize_to_palette(image, palette)


def quantize_to_palette(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    flat = image.reshape((-1, 3)).astype(np.float32)
    distances = np.sum((flat[:, None, :] - palette[None, :, :]) ** 2, axis=2)
    return palette[np.argmin(distances, axis=1)].reshape(image.shape).astype(np.uint8)


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
    edges = cv2.Canny(gray, 50, 120)
    edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8), iterations=1)
    result = image.copy()
    result[edges > 0] = (0, 0, 0)
    return result


def _analysis_log(analysis: ImageAnalysis, elapsed: float, output_size: Tuple[int, int]) -> None:
    print(
        "PixelArt debug | "
        f"input={analysis.width}x{analysis.height} aspect={analysis.aspect_ratio:.3f} "
        f"complexity={analysis.complexity:.3f} pixel_size={analysis.chosen_pixel_size} "
        f"grid={analysis.grid_size[0]}x{analysis.grid_size[1]} palette={analysis.palette_size} "
        f"dominant={analysis.dominant_colors} time={elapsed:.2f}s output={output_size[0]}x{output_size[1]}"
    )



def build_pixel_art_prompt(options: ProcessingOptions, analysis: ImageAnalysis) -> str:
    style_name = STYLE_NAMES.get(options.style, options.style)
    custom = f", {options.ai_prompt.strip()}" if options.ai_prompt.strip() else ""
    return (
        f"professional {style_name} pixel art, {options.quality} quality, 8-bit and 16-bit game asset, "
        f"RPG sprite style, clean black outlines, limited {options.palette_size} color palette, detailed cel shading, "
        f"crisp square pixels, preserve original character pose and composition, aspect ratio {analysis.aspect_ratio:.2f}{custom}"
    )


def prepare_ai_source(input_file: str, output_file: str, options: ProcessingOptions) -> Tuple[str, str]:
    image, alpha = _read_image(input_file)
    analysis = analyze_image_array(image, options.pixel_size, options.palette_size)
    block = max(4, min(16, analysis.chosen_pixel_size))
    grid_w = max(32, min(128, analysis.width // block))
    grid_h = max(32, min(128, analysis.height // block))
    preview_options = ProcessingOptions(
        style=options.style, pixel_size=max(1, analysis.width // grid_w), palette_size=options.palette_size,
        quality="hd", outline="thin", dithering=True, sharpen=True, generation_mode="classic"
    )
    process_image(input_file, output_file, preview_options)
    return output_file, build_pixel_art_prompt(options, analysis)


def generate_ai_pixel_art(input_file: str, output_file: str, options: ProcessingOptions) -> str:
    endpoint = os.getenv("STABLE_DIFFUSION_API_URL", "").strip()
    api_key = os.getenv("STABLE_DIFFUSION_API_KEY", "").strip()
    prepared_file = output_file.replace(".png", "_ai_source.png")
    source_file, prompt = prepare_ai_source(input_file, prepared_file, options)
    if not endpoint:
        print("AI Pixel debug | STABLE_DIFFUSION_API_URL is not configured; returning AI-ready source image")
        Image.open(source_file).save(output_file)
        return output_file

    with open(source_file, "rb") as image_fp:
        init_image = base64.b64encode(image_fp.read()).decode("ascii")
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, smooth gradients, photorealistic, anti-aliased, noisy, deformed, extra limbs",
        "init_images": [init_image],
        "denoising_strength": options.ai_strength,
        "steps": options.ai_steps,
        "cfg_scale": 8,
        "sampler_name": "DPM++ 2M Karras",
    }
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Stable Diffusion API request failed: {exc}") from exc
    images = data.get("images") or data.get("artifacts") or []
    if not images:
        raise RuntimeError("Stable Diffusion API response did not contain generated images")
    encoded = images[0].get("base64") if isinstance(images[0], dict) else images[0]
    if isinstance(encoded, str) and "," in encoded and encoded.startswith("data:image"):
        encoded = encoded.split(",", 1)[1]
    Image.open(io.BytesIO(base64.b64decode(encoded))).save(output_file)
    return output_file

def process_image(input_file: str, output_file: str, options: ProcessingOptions) -> str:
    if options.generation_mode == "ai":
        return generate_ai_pixel_art(input_file, output_file, options)

    start = time.perf_counter()
    image, alpha = _read_image(input_file)
    h, w = image.shape[:2]
    max_side = 1600
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, new_size, interpolation=cv2.INTER_AREA) if alpha is not None else None
        h, w = image.shape[:2]

    analysis = analyze_image_array(image, options.pixel_size, options.palette_size)
    block = analysis.chosen_pixel_size
    grid_w, grid_h = analysis.grid_size
    quality = options.quality if options.quality in QUALITY_MODES else ("hd" if options.style == "hd" else "normal")

    work = gamma_correct(image, options.gamma)
    if quality != "fast":
        work = apply_auto_contrast(work)
        if quality == "hd":
            work = apply_clahe(work)
        if options.denoise or quality == "hd":
            den = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(work, cv2.COLOR_RGB2BGR), None, 4, 4, 7, 21)
            work = cv2.cvtColor(den, cv2.COLOR_BGR2RGB)
        if options.sharpen or quality == "hd":
            work = sharpen(work)

    # Build the real pixel grid first, then quantize that grid so the output contains hard-edged blocks.
    small = cv2.resize(work, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    if quality == "fast":
        palette = _palette_array(options.style, options.palette_size)
        small = quantize_to_palette(small, palette)
    else:
        learned_palette = kmeans_palette(small, options.palette_size)
        style_palette = _palette_array(options.style, options.palette_size)
        palette = learned_palette if options.style in ("classic", "hd") else style_palette
        small = floyd_steinberg(small, palette) if options.dithering or quality == "hd" else quantize_to_palette(small, palette)
    small = add_outline(small, OUTLINE_LEVELS.get(options.outline, 0) if (options.outline != "off" or quality == "hd") else 0)

    result = cv2.resize(small, (grid_w * block, grid_h * block), interpolation=cv2.INTER_NEAREST)
    if alpha is not None and np.min(alpha) < 255:
        alpha_small = cv2.resize(alpha, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        alpha_out = cv2.resize(alpha_small, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = np.dstack([result, alpha_out])
        Image.fromarray(out, "RGBA").save(output_file)
    else:
        Image.fromarray(result).save(output_file)
    _analysis_log(analysis, time.perf_counter() - start, (result.shape[1], result.shape[0]))
    return output_file


def pixel_art(input_file, output_file, pixel_size=16):
    return process_image(input_file, output_file, ProcessingOptions(pixel_size=pixel_size))


def create_comparison(input_file: str, processed_file: str, output_file: str) -> str:
    before = Image.open(input_file).convert("RGB")
    after = Image.open(processed_file).convert("RGB")
    before.thumbnail((700, 700), Image.Resampling.LANCZOS)
    after.thumbnail((700, 700), Image.Resampling.NEAREST)
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
    image, _ = _read_image(path)
    analysis = analyze_image_array(image, None, 16)
    with Image.open(path) as img:
        stat = os.stat(path)
        return {"format": img.format or "Unknown", "size": f"{img.width}×{img.height}", "mode": img.mode, "file_size": f"{stat.st_size / 1024:.1f} KB", "aspect_ratio": f"{analysis.aspect_ratio:.3f}", "complexity": f"{analysis.complexity:.3f}", "auto_pixel_size": str(analysis.chosen_pixel_size), "grid_size": f"{analysis.grid_size[0]}×{analysis.grid_size[1]}", "dominant_colors": str(analysis.dominant_colors)}


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
