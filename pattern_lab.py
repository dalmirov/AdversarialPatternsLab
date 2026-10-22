"""Model-independent utilities for the local textile Pattern Lab."""
from functools import lru_cache
import hashlib

import cv2
import numpy as np

from pattern_library import REGISTRY, generate

APP_VERSION = "0.1.0"
DEFAULT_SEED = 0
DEFAULT_SOURCE_SIZE = 300
MIN_SOURCE_SIZE = 32
MAX_SOURCE_SIZE = 1200
MIN_TILE_SIZE = 24
MAX_TILE_SIZE = 600
PREVIEW_EDGE = 720

CATEGORY_LABELS = {
    "baseline_geometric": "Geométricos",
    "natural_texture": "Texturas naturales",
    "object_shapes": "Formas y objetos",
    "camouflage": "Camuflaje",
    "optimisation_inspired": "Ópticos y alta frecuencia",
    "decoy": "Elementos figurativos",
}


def validate_pattern_name(name):
    if name not in REGISTRY:
        raise ValueError("El patrón solicitado no existe en la biblioteca.")
    return name


def validate_seed(seed):
    try:
        value = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("La semilla debe ser un número entero.") from exc
    if value < 0 or value > 2_147_483_647:
        raise ValueError("La semilla debe estar entre 0 y 2147483647.")
    return value


def validate_size(size, minimum=MIN_SOURCE_SIZE, maximum=MAX_SOURCE_SIZE, field="tamaño"):
    try:
        value = int(size)
    except (TypeError, ValueError) as exc:
        raise ValueError(field.capitalize() + " debe ser un número entero.") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{field.capitalize()} debe estar entre {minimum} y {maximum} píxeles.")
    return value


@lru_cache(maxsize=256)
def _cached_patch(name, seed, size):
    image = generate(name, np.random.default_rng(seed), size)
    image.setflags(write=False)
    return image


def generate_patch(name, seed=DEFAULT_SEED, size=DEFAULT_SOURCE_SIZE):
    """Return a reproducible BGR patch generated from an existing registry entry."""
    name = validate_pattern_name(name)
    seed = validate_seed(seed)
    size = validate_size(size, field="tamaño base")
    return _cached_patch(name, seed, size).copy()


def resize_patch(patch, tile_size):
    tile_size = validate_size(tile_size, MIN_TILE_SIZE, MAX_TILE_SIZE, "tamaño del motivo")
    source_h, source_w = patch.shape[:2]
    interpolation = cv2.INTER_AREA if tile_size < min(source_h, source_w) else cv2.INTER_LINEAR
    return cv2.resize(patch, (tile_size, tile_size), interpolation=interpolation)


def repeated_preview(patch, tile_size, edge=PREVIEW_EDGE):
    """Create a square seamless-grid preview with a controllable motif size."""
    tile_size = validate_size(tile_size, MIN_TILE_SIZE, MAX_TILE_SIZE, "tamaño del motivo")
    edge = validate_size(edge, 64, 1600, "tamaño de la vista")
    tile = resize_patch(patch, tile_size)
    rows = (edge + tile_size - 1) // tile_size
    cols = (edge + tile_size - 1) // tile_size
    return np.tile(tile, (rows, cols, 1))[:edge, :edge].copy()


def get_catalog():
    items = []
    for name, spec in REGISTRY.items():
        items.append({
            "name": name,
            "label": name.replace("_", " ").title(),
            "category": spec.category,
            "category_label": CATEGORY_LABELS.get(spec.category, spec.category.replace("_", " ").title()),
            "rationale": spec.rationale,
            "reference": spec.reference,
        })
    categories = list(dict.fromkeys(spec.category for spec in REGISTRY.values()))
    return {
        "app": "Pattern Lab",
        "version": APP_VERSION,
        "count": len(items),
        "categories": [
            {"id": category, "label": CATEGORY_LABELS.get(category, category.replace("_", " ").title())}
            for category in categories
        ],
        "patterns": items,
    }


def variant_metadata(name, seed=DEFAULT_SEED, size=DEFAULT_SOURCE_SIZE, tile_size=160):
    name = validate_pattern_name(name)
    seed = validate_seed(seed)
    size = validate_size(size, field="tamaño base")
    tile_size = validate_size(tile_size, MIN_TILE_SIZE, MAX_TILE_SIZE, "tamaño del motivo")
    patch = generate_patch(name, seed, size)
    spec = REGISTRY[name]
    return {
        "app": "Pattern Lab",
        "app_version": APP_VERSION,
        "pattern": name,
        "pattern_label": name.replace("_", " ").title(),
        "category": spec.category,
        "category_label": CATEGORY_LABELS.get(spec.category, spec.category.replace("_", " ").title()),
        "seed": seed,
        "source_size_px": size,
        "display_tile_size_px": tile_size,
        "display_scale": round(tile_size / size, 4),
        "sha256_pixels_bgr": hashlib.sha256(np.ascontiguousarray(patch).tobytes()).hexdigest(),
        "reproducibility": {
            "generator": "pattern_library.generate",
            "seed": seed,
            "size": size,
            "note": "El mismo código, patrón, semilla y tamaño base reproducen los mismos píxeles en el mismo entorno de dependencias.",
        },
        "rationale": spec.rationale,
        "reference": spec.reference,
        "license": "GPL-3.0",
    }


def encode_png(image):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen PNG.")
    return buffer.tobytes()
