# Adversarial Pattern Lab
# Roni Bandini, Oct 2026, GPL License
# Pattern library


from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


# 8 patrones originaales

def _grid(size):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    return x, y


def _palette(rng, count):
    return rng.integers(0, 256, (count, 3), dtype=np.uint8)


def p_blocks(rng, size):
    block = int(rng.integers(6, 40))
    n = -(-size // block)
    palette = _palette(rng, int(rng.integers(2, 6)))
    img = palette[rng.integers(0, len(palette), (n, n))]
    return np.repeat(np.repeat(img, block, 0), block, 1)[:size, :size]


def p_stripes(rng, size):
    x, y = _grid(size)
    angle, frequency = rng.uniform(0, np.pi), rng.uniform(0.05, 0.5)
    index = (np.sin((x * np.cos(angle) + y * np.sin(angle)) * frequency) > 0).astype(int)
    return _palette(rng, 2)[index]


def p_checker(rng, size):
    x, y = _grid(size)
    block = int(rng.integers(8, 50))
    index = ((x // block + y // block) % 2).astype(int)
    return _palette(rng, 2)[index]


def p_rings(rng, size):
    x, y = _grid(size)
    cx, cy = rng.uniform(0, size, 2)
    frequency = rng.uniform(0.1, 0.6)
    index = (np.sin(np.hypot(x - cx, y - cy) * frequency) > 0).astype(int)
    return _palette(rng, 2)[index]


def p_smooth(rng, size):
    count = int(rng.integers(3, 12))
    low = rng.random((count, count, 3)).astype(np.float32)
    img = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def p_voronoi(rng, size):
    count = int(rng.integers(6, 40))
    points = rng.uniform(0, size, (count, 2))
    x, y = _grid(size)
    distances = (x[..., None] - points[:, 0]) ** 2 + (y[..., None] - points[:, 1]) ** 2
    return _palette(rng, count)[distances.argmin(-1)]


def p_waves(rng, size):
    x, y = _grid(size)
    field = np.zeros((size, size), np.float32)
    for _ in range(int(rng.integers(2, 6))):
        angle, frequency, phase = rng.uniform(0, 2 * np.pi), rng.uniform(0.03, 0.3), rng.uniform(0, 2 * np.pi)
        field += np.sin((x * np.cos(angle) + y * np.sin(angle)) * frequency + phase)
    field = (field - field.min()) / (np.ptp(field) + 1e-6)
    hsv = np.stack([(field * 179).astype(np.uint8), np.full((size, size), 255, np.uint8),
                    np.full((size, size), 255, np.uint8)], -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def p_noise(rng, size):
    block = int(rng.integers(1, 5))
    n = -(-size // block)
    img = rng.integers(0, 256, (n, n, 3), dtype=np.uint8)
    return np.repeat(np.repeat(img, block, 0), block, 1)[:size, :size]


BASE_PATTERNS = {"blocks": p_blocks, "stripes": p_stripes, "checker": p_checker,
                 "rings": p_rings, "smooth": p_smooth, "voronoi": p_voronoi,
                 "waves": p_waves, "noise": p_noise}


@dataclass(frozen=True)
class PatternSpec:
    name: str
    fn: Callable
    category: str
    rationale: str
    reference: str = ""


def _t(size, base=1.0):
    """Line thickness scaled to canvas size."""
    return max(1, int(round(base * size / 300)))


def _bright(rng):
    hsv = np.uint8([[[rng.integers(0, 180), rng.integers(200, 256), rng.integers(200, 256)]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _rand_color(rng):
    return tuple(int(c) for c in rng.integers(0, 256, 3))


def _jitter(color, rng, amount=18):
    return tuple(int(np.clip(c + rng.integers(-amount, amount + 1), 0, 255)) for c in color)


def _smooth_noise(rng, size, cells):
    low = rng.random((cells, cells)).astype(np.float32)
    return cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)


def _warp(img, rng, amp=0.05, cells=4):
    size = img.shape[0]
    dx = cv2.resize(rng.uniform(-1, 1, (cells, cells)).astype(np.float32), (size, size),
                    interpolation=cv2.INTER_CUBIC) * amp * size
    dy = cv2.resize(rng.uniform(-1, 1, (cells, cells)).astype(np.float32), (size, size),
                    interpolation=cv2.INTER_CUBIC) * amp * size
    gx, gy = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    return cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _grid_centres(rng, size, cell, stagger=True, jitter=0.0):
    n = int(np.ceil(size / cell)) + 1
    for r in range(-1, n):
        for c in range(-1, n):
            x = c * cell + (cell / 2 if stagger and r % 2 else 0)
            y = r * cell
            if jitter:
                x += rng.uniform(-jitter, jitter) * cell
                y += rng.uniform(-jitter, jitter) * cell
            yield int(x), int(y), r, c


# Texturas
def p_leopard(rng, size):
    base = _jitter((90, 170, 220), rng)  # tan (BGR)
    img = np.full((size, size, 3), base, np.uint8)
    cell = size / rng.uniform(5, 8)
    for x, y, _, _ in _grid_centres(rng, size, cell, jitter=0.25):
        rad = cell * rng.uniform(0.28, 0.4)
        cv2.circle(img, (x, y), int(rad * 0.8), _jitter((50, 110, 170), rng), -1, cv2.LINE_AA)
        for k in range(int(rng.integers(5, 8))):
            a = 2 * np.pi * k / 6 + rng.uniform(-0.3, 0.3)
            cx, cy = x + rad * np.cos(a), y + rad * np.sin(a)
            cv2.ellipse(img, (int(cx), int(cy)), (int(rad * 0.38), int(rad * 0.24)),
                        np.degrees(a), 0, 360, (15, 20, 25), -1, cv2.LINE_AA)
    return img


def p_zebra(rng, size):
    x, y = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    angle = rng.uniform(0, np.pi)
    freq = rng.uniform(0.06, 0.2)
    wobble = 4 * np.sin(y * 0.03 + rng.uniform(0, 6)) * np.cos(x * 0.02 + rng.uniform(0, 6))
    field = np.sin((x * np.cos(angle) + y * np.sin(angle)) * freq + wobble)
    img = np.where(field[..., None] > 0, 255, 15).astype(np.uint8).repeat(3, -1)
    return _warp(img, rng, amp=0.06)


def p_giraffe(rng, size):
    res = 256
    count = int(rng.integers(22, 40))
    pts = rng.uniform(0, res, (count, 2)).astype(np.float32)
    gx, gy = np.meshgrid(np.arange(res, dtype=np.float32), np.arange(res, dtype=np.float32))
    d = np.stack([(gx - p[0]) ** 2 + (gy - p[1]) ** 2 for p in pts], -1)
    order = np.sort(d, -1)
    edge = (np.sqrt(order[..., 1]) - np.sqrt(order[..., 0])) < res * 0.018
    shades = np.array([_jitter((40, 100, 160), rng, 25) for _ in range(count)], np.uint8)
    img = shades[d.argmin(-1)]
    img[edge] = (200, 225, 240)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def p_cow(rng, size):
    noise = _smooth_noise(rng, size, int(rng.integers(4, 8)))
    return np.where(noise[..., None] > np.quantile(noise, 0.55), 20, 245).astype(np.uint8).repeat(3, -1)


def p_fish_scales(rng, size):
    d = size / rng.uniform(8, 16)
    base = _bright(rng)
    img = np.zeros((size, size, 3), np.uint8)
    for row in range(int(size / (d / 2)) + 3):
        for col in range(int(size / d) + 3):
            c = (int(col * d + (row % 2) * d / 2), int(row * d / 2))
            tint = _jitter(base, rng, 35)
            cv2.circle(img, c, int(d * 0.62), tint, -1, cv2.LINE_AA)
            cv2.circle(img, c, int(d * 0.62), (20, 20, 20), _t(size), cv2.LINE_AA)
            cv2.circle(img, c, int(d * 0.3), _jitter(tint, rng, 40), _t(size), cv2.LINE_AA)
    return img


def p_fur(rng, size):
    img = np.full((size, size, 3), _rand_color(rng), np.uint8)
    palette = [_rand_color(rng) for _ in range(4)]
    length = size * 0.06
    for _ in range(int(size * size / 40)):
        x, y = rng.uniform(0, size, 2)
        a = rng.normal(1.2, 0.35)
        cv2.line(img, (int(x), int(y)), (int(x + length * np.cos(a)), int(y + length * np.sin(a))),
                 palette[int(rng.integers(0, 4))], _t(size), cv2.LINE_AA)
    return img


def p_flowers(rng, size):
    img = np.full((size, size, 3), _jitter((60, 120, 60), rng), np.uint8)
    cell = size / rng.uniform(3, 5)
    for x, y, _, _ in _grid_centres(rng, size, cell, jitter=0.15):
        petal, centre = _bright(rng), _bright(rng)
        n = int(rng.integers(5, 9))
        for k in range(n):
            a = 2 * np.pi * k / n
            c = (int(x + cell * 0.22 * np.cos(a)), int(y + cell * 0.22 * np.sin(a)))
            cv2.ellipse(img, c, (int(cell * 0.2), int(cell * 0.1)), np.degrees(a), 0, 360, petal, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), int(cell * 0.1), centre, -1, cv2.LINE_AA)
    return img


# Objetos
def p_eyes(rng, size):
    img = np.full((size, size, 3), _rand_color(rng), np.uint8)
    cell = size / rng.uniform(3, 6)
    for x, y, _, _ in _grid_centres(rng, size, cell, jitter=0.1):
        w, h = int(cell * 0.38), int(cell * rng.uniform(0.16, 0.3))
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (240, 240, 240), -1, cv2.LINE_AA)
        iris = _bright(rng)
        cv2.circle(img, (x, y), int(h * 0.9), iris, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, y), (max(1, int(h * 0.25)), int(h * 0.8)), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, (10, 10, 10), _t(size, 2), cv2.LINE_AA)
    return img


def p_faces(rng, size):
    img = np.full((size, size, 3), _rand_color(rng), np.uint8)
    cell = size / rng.uniform(3, 6)
    for x, y, _, _ in _grid_centres(rng, size, cell, jitter=0.1):
        r = int(cell * 0.4)
        cv2.circle(img, (x, y), r, _jitter((120, 180, 230), rng, 30), -1, cv2.LINE_AA)
        for sx in (-1, 1):
            cv2.circle(img, (int(x + sx * r * 0.35), int(y - r * 0.2)), max(1, int(r * 0.12)), (20, 20, 20), -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, int(y + r * 0.2)), (int(r * 0.4), int(r * 0.25)), 0, 0, 180, (20, 20, 120), _t(size, 2), cv2.LINE_AA)
    return img


def p_octagons(rng, size):
    img = np.full((size, size, 3), (230, 230, 230), np.uint8)
    cell = size / rng.uniform(2.5, 4.5)
    for x, y, _, _ in _grid_centres(rng, size, cell, jitter=0.05):
        r = cell * 0.46
        pts = np.array([[x + r * np.cos(np.pi / 8 + k * np.pi / 4), y + r * np.sin(np.pi / 8 + k * np.pi / 4)]
                        for k in range(8)], np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255), cv2.LINE_AA)
        cv2.fillPoly(img, [((pts - [x, y]) * 0.88 + [x, y]).astype(np.int32)], (40, 40, 200), cv2.LINE_AA)
    return img


def p_traffic_lights(rng, size):
    img = np.full((size, size, 3), _jitter((70, 70, 70), rng), np.uint8)
    cell = size / rng.uniform(3, 5)
    cols = [(40, 40, 230), (40, 200, 240), (60, 200, 60)]
    for x, y, _, _ in _grid_centres(rng, size, cell, stagger=False):
        w, h = int(cell * 0.2), int(cell * 0.42)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (15, 15, 15), -1)
        lit = int(rng.integers(0, 3))
        for k in range(3):
            c = cols[k] if k == lit else (45, 45, 45)
            cv2.circle(img, (x, int(y + (k - 1) * h * 0.62)), int(w * 0.68), c, -1, cv2.LINE_AA)
    return img


def p_wheels(rng, size):
    img = np.full((size, size, 3), _rand_color(rng), np.uint8)
    cell = size / rng.uniform(2.5, 4.5)
    for x, y, _, _ in _grid_centres(rng, size, cell, jitter=0.1):
        r = int(cell * 0.44)
        cv2.circle(img, (x, y), r, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), int(r * 0.7), (200, 200, 200), _t(size, 2), cv2.LINE_AA)
        off = rng.uniform(0, np.pi)
        for k in range(int(rng.integers(6, 12))):
            a = off + 2 * np.pi * k / 8
            cv2.line(img, (x, y), (int(x + r * 0.7 * np.cos(a)), int(y + r * 0.7 * np.sin(a))),
                     (200, 200, 200), _t(size), cv2.LINE_AA)
        cv2.circle(img, (x, y), max(2, int(r * 0.12)), (200, 200, 200), -1, cv2.LINE_AA)
    return img


def p_clocks(rng, size):
    img = np.full((size, size, 3), _rand_color(rng), np.uint8)
    cell = size / rng.uniform(2.5, 4.5)
    for x, y, _, _ in _grid_centres(rng, size, cell, jitter=0.05):
        r = cell * 0.44
        cv2.circle(img, (x, y), int(r), (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), int(r), (20, 20, 20), _t(size, 2), cv2.LINE_AA)
        for k in range(12):
            a = 2 * np.pi * k / 12
            cv2.line(img, (int(x + r * 0.82 * np.cos(a)), int(y + r * 0.82 * np.sin(a))),
                     (int(x + r * 0.95 * np.cos(a)), int(y + r * 0.95 * np.sin(a))), (20, 20, 20), _t(size), cv2.LINE_AA)
        for length, th in ((0.55, 2), (0.8, 1)):
            a = rng.uniform(0, 2 * np.pi)
            cv2.line(img, (x, y), (int(x + r * length * np.cos(a)), int(y + r * length * np.sin(a))),
                     (20, 20, 20), _t(size, th), cv2.LINE_AA)
    return img


def p_vehicles(rng, size):
    img = np.full((size, size, 3), (120, 120, 120), np.uint8)
    cell = size / rng.uniform(2.5, 4)
    for x, y, _, _ in _grid_centres(rng, size, cell, jitter=0.05):
        w, h = cell * 0.42, cell * 0.14
        body = _bright(rng)
        cv2.rectangle(img, (int(x - w), int(y - h * 0.2)), (int(x + w), int(y + h)), body, -1, cv2.LINE_AA)
        cabin = np.array([[x - w * 0.55, y - h * 0.2], [x - w * 0.3, y - h * 1.3],
                          [x + w * 0.35, y - h * 1.3], [x + w * 0.65, y - h * 0.2]], np.int32)
        cv2.fillPoly(img, [cabin], body, cv2.LINE_AA)
        cv2.fillPoly(img, [((cabin - [x, y]) * 0.7 + [x, y - h * 0.1]).astype(np.int32)], (220, 200, 160), cv2.LINE_AA)
        for sx in (-0.55, 0.55):
            cv2.circle(img, (int(x + sx * w), int(y + h)), int(h * 0.55), (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x + sx * w), int(y + h)), int(h * 0.25), (190, 190, 190), -1, cv2.LINE_AA)
    return img


def p_glyphs(rng, size):
    img = np.full((size, size, 3), _rand_color(rng), np.uint8)
    chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    scale = size / 300
    for _ in range(int(rng.integers(12, 30))):
        ch = chars[int(rng.integers(0, len(chars)))]
        cv2.putText(img, ch, (int(rng.uniform(-10, size)), int(rng.uniform(10, size + 10))),
                    cv2.FONT_HERSHEY_DUPLEX, rng.uniform(1.5, 4.5) * scale, _bright(rng),
                    _t(size, rng.integers(3, 8)), cv2.LINE_AA)
    return img


def p_barcode(rng, size):
    img = np.full((size, size, 3), 255, np.uint8)
    x = 0
    while x < size:
        w = int(rng.integers(1, 6)) * _t(size, 2)
        if rng.random() < 0.5:
            img[:, x:x + w] = 0
        x += w
    if rng.random() < 0.5:
        img = img.transpose(1, 0, 2).copy()
    return img


def p_qr_like(rng, size):
    n = int(rng.integers(21, 33))
    bits = (rng.random((n, n)) > 0.5).astype(np.uint8) * 255
    for oy, ox in ((0, 0), (0, n - 7), (n - 7, 0)):
        bits[oy:oy + 7, ox:ox + 7] = 0
        bits[oy + 1:oy + 6, ox + 1:ox + 6] = 255
        bits[oy + 2:oy + 5, ox + 2:ox + 5] = 0
    return cv2.resize(bits, (size, size), interpolation=cv2.INTER_NEAREST)[..., None].repeat(3, -1)


def p_stick_figures(rng, size):
    """Decoy category: person-like cues. Could RAISE detection, which is also worth measuring."""
    img = np.full((size, size, 3), _rand_color(rng), np.uint8)
    cell = size / rng.uniform(4, 7)
    for x, y, _, _ in _grid_centres(rng, size, cell, jitter=0.1):
        col, th = _bright(rng), _t(size, 2)
        s = cell * 0.4
        cv2.circle(img, (x, int(y - s * 0.8)), int(s * 0.22), col, -1, cv2.LINE_AA)
        cv2.line(img, (x, int(y - s * 0.55)), (x, int(y + s * 0.25)), col, th, cv2.LINE_AA)
        for sx in (-1, 1):
            cv2.line(img, (x, int(y - s * 0.4)), (int(x + sx * s * 0.45), int(y + s * 0.05)), col, th, cv2.LINE_AA)
            cv2.line(img, (x, int(y + s * 0.25)), (int(x + sx * s * 0.3), int(y + s * 0.9)), col, th, cv2.LINE_AA)
    return img


# Camuflado
def p_dazzle(rng, size):
    img = np.full((size, size, 3), 255, np.uint8)
    x, y = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    for _ in range(int(rng.integers(10, 18))):
        pts = np.array([[rng.uniform(0, size), rng.uniform(0, size)] for _ in range(int(rng.integers(3, 6)))], np.int32)
        a, f = rng.uniform(0, np.pi), rng.uniform(0.08, 0.35)
        stripes = np.where(np.sin((x * np.cos(a) + y * np.sin(a)) * f) > 0, 0, 255).astype(np.uint8)[..., None].repeat(3, -1)
        m = np.zeros((size, size), np.uint8)
        cv2.fillPoly(m, [pts], 255)
        img[m > 0] = stripes[m > 0]
    return img


def p_digital_camo(rng, size):
    palette = [(40, 60, 70), (90, 120, 110), (60, 100, 85), (130, 160, 150)]
    res = 48
    img = np.full((res, res, 3), palette[0], np.uint8)
    for col in palette[1:]:
        noise = cv2.resize(rng.random((int(rng.integers(4, 9)),) * 2).astype(np.float32), (res, res),
                           interpolation=cv2.INTER_CUBIC)
        img[noise > 0.55] = col
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def p_woodland_camo(rng, size):
    palette = np.array([(35, 55, 60), (50, 90, 70), (70, 120, 90), (45, 70, 110)], np.uint8)
    noise = _smooth_noise(rng, size, int(rng.integers(5, 9)))
    q = np.digitize(noise, np.quantile(noise, [0.25, 0.5, 0.75]))
    return palette[q]


# Optimizados
def p_spiral(rng, size):
    x, y = np.meshgrid(np.linspace(-1, 1, size, dtype=np.float32), np.linspace(-1, 1, size, dtype=np.float32))
    theta, r = np.arctan2(y, x), np.hypot(x, y)
    h = ((theta / (2 * np.pi) * int(rng.integers(2, 7)) + r * rng.uniform(1, 5)) % 1.0)
    hsv = np.stack([(h * 179).astype(np.uint8), np.full((size, size), 255, np.uint8), np.full((size, size), 255, np.uint8)], -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def p_blob_saturated(rng, size):
    noise = _smooth_noise(rng, size, int(rng.integers(4, 9)))
    levels = int(rng.integers(4, 8))
    q = np.floor((noise - noise.min()) / (np.ptp(noise) + 1e-6) * levels).clip(0, levels - 1)
    hsv = np.stack([((q / levels * 179 + rng.integers(0, 179)) % 179).astype(np.uint8),
                    np.full((size, size), 255, np.uint8), np.full((size, size), 255, np.uint8)], -1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    edges = cv2.Canny((q * 255 / levels).astype(np.uint8), 1, 2)
    img[cv2.dilate(edges, np.ones((_t(size, 2),) * 2, np.uint8)) > 0] = 0
    return img


def p_grating_multi(rng, size):
    x, y = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    chans = []
    for _ in range(3):
        a, f, ph = rng.uniform(0, np.pi), rng.uniform(0.15, 0.9), rng.uniform(0, 2 * np.pi)
        chans.append(((np.sin((x * np.cos(a) + y * np.sin(a)) * f + ph) > 0) * 255).astype(np.uint8))
    return np.stack(chans, -1)


def p_bullseye(rng, size):
    img = np.full((size, size, 3), _rand_color(rng), np.uint8)
    cell = size / rng.uniform(2, 4)
    palette = [_bright(rng) for _ in range(4)]
    for x, y, _, _ in _grid_centres(rng, size, cell, jitter=0.05):
        for k, r in enumerate(np.linspace(cell * 0.46, cell * 0.05, 6)):
            cv2.circle(img, (x, y), int(r), palette[k % 4], -1, cv2.LINE_AA)
    return img


# Geometricos
def p_polka(rng, size):
    a, b = _rand_color(rng), _bright(rng)
    img = np.full((size, size, 3), a, np.uint8)
    cell = size / rng.uniform(5, 12)
    for x, y, _, _ in _grid_centres(rng, size, cell):
        cv2.circle(img, (x, y), int(cell * rng.uniform(0.2, 0.38)), b, -1, cv2.LINE_AA)
    return img


def p_hexagons(rng, size):
    pal = [_rand_color(rng) for _ in range(int(rng.integers(2, 5)))]
    img = np.zeros((size, size, 3), np.uint8)
    r = size / rng.uniform(5, 12)
    for row in range(int(size / (r * 1.5)) + 3):
        for col in range(int(size / (r * 1.74)) + 3):
            cx, cy = col * r * 1.732 + (row % 2) * r * 0.866, row * r * 1.5
            pts = np.array([[cx + r * np.cos(np.pi / 6 + k * np.pi / 3), cy + r * np.sin(np.pi / 6 + k * np.pi / 3)]
                            for k in range(6)], np.int32)
            cv2.fillPoly(img, [pts], pal[int(rng.integers(0, len(pal)))], cv2.LINE_AA)
            cv2.polylines(img, [pts], True, (15, 15, 15), _t(size), cv2.LINE_AA)
    return img


def p_chevron(rng, size):
    a, b = _rand_color(rng), _rand_color(rng)
    img = np.full((size, size, 3), a, np.uint8)
    step = int(size / rng.uniform(6, 14))
    amp = step * rng.uniform(0.5, 1.5)
    xs = np.arange(0, size + int(amp * 2), int(max(2, amp)))
    for y0 in range(-step, size + step, step * 2):
        pts = [(int(x), int(y0 + (amp if i % 2 else 0))) for i, x in enumerate(xs)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, b, step, cv2.LINE_AA)
    return img


def _spec(name, fn, cat, why, ref=""):
    return PatternSpec(name, fn, cat, why, ref)


_TEXTURE_BIAS = ("Geirhos et al., 'ImageNet-trained CNNs are biased towards texture', ICLR 2019")
_ADV_PATCH = "Brown et al., 'Adversarial Patch', 2017"
_THYS = "Thys et al., 'Fooling automated surveillance cameras', CVPRW 2019"
_TSHIRT = "Xu et al., 'Adversarial T-shirts', ECCV 2020"
_TEXTURE = "Hu et al., 'Adversarial Texture for Fooling Person Detectors', CVPR 2022"
_NATURAL = "Hu et al., 'Naturalistic Physical Adversarial Patch for Object Detectors', ICCV 2021"

SPECS = [
    # baselines already in shirt_pattern_loop.py
    *[_spec(n, f, "baseline_geometric", "Original generator: unstructured geometric/noise controls.") for n, f in BASE_PATTERNS.items()],
    _spec("polka", p_polka, "baseline_geometric", "Regular dots; periodic control.", _TEXTURE),
    _spec("hexagons", p_hexagons, "baseline_geometric", "Honeycomb; repeated cells like tiled textures.", _TEXTURE),
    _spec("chevron", p_chevron, "baseline_geometric", "Zig-zag edges with strong gradients.", ""),
    # natural textures
    _spec("leopard", p_leopard, "natural_texture", "Animal-coat texture; CNNs lean on texture, so fur cues may shift detections.", _TEXTURE_BIAS),
    _spec("zebra", p_zebra, "natural_texture", "Warped high-contrast stripes that break body contour cues.", _TEXTURE_BIAS),
    _spec("giraffe", p_giraffe, "natural_texture", "Patchwork cells with bright borders.", _TEXTURE_BIAS),
    _spec("cow", p_cow, "natural_texture", "Large blotches that cut across limb boundaries.", _TEXTURE_BIAS),
    _spec("fish_scales", p_fish_scales, "natural_texture", "Dense overlapping arcs, high local contrast.", _TEXTURE_BIAS),
    _spec("fur", p_fur, "natural_texture", "Directional strokes; high-frequency texture.", _TEXTURE_BIAS),
    _spec("flowers", p_flowers, "natural_texture", "Saturated radial motifs on a dark ground.", _NATURAL),
    # object shapes
    _spec("eyes", p_eyes, "object_shapes", "Animal-eye motifs; naturalistic patches with animal faces were used against detectors.", _NATURAL),
    _spec("faces", p_faces, "object_shapes", "Face-like cues; faces are a strong trigger for some detectors.", _NATURAL),
    _spec("octagons", p_octagons, "object_shapes", "Generic sign-shaped polygons (other object class).", _ADV_PATCH),
    _spec("traffic_lights", p_traffic_lights, "object_shapes", "Stacked lit discs; non-person object class.", _ADV_PATCH),
    _spec("wheels", p_wheels, "object_shapes", "Radial spokes; a strong non-person shape.", _ADV_PATCH),
    _spec("clocks", p_clocks, "object_shapes", "Circular dials with ticks.", _ADV_PATCH),
    _spec("vehicles", p_vehicles, "object_shapes", "Simplified car silhouettes (a competing detector class).", _ADV_PATCH),
    _spec("glyphs", p_glyphs, "object_shapes", "Large text characters, saturated colours.", ""),
    _spec("barcode", p_barcode, "object_shapes", "1-D high-contrast bars.", ""),
    _spec("qr_like", p_qr_like, "object_shapes", "2-D binary modules, very high frequency.", ""),
    # camouflage
    _spec("dazzle", p_dazzle, "camouflage", "Dazzle-style overlapping stripe polygons; breaks outlines (evidence is strongest for face detectors).", ""),
    _spec("digital_camo", p_digital_camo, "camouflage", "Pixelated blotches, low contrast with background.", ""),
    _spec("woodland_camo", p_woodland_camo, "camouflage", "Organic blotches.", ""),
    # optimisation-inspired
    _spec("spiral", p_spiral, "optimisation_inspired", "Saturated swirls resemble the look of universal adversarial patches (but are not optimised).", _ADV_PATCH),
    _spec("blob_saturated", p_blob_saturated, "optimisation_inspired", "Saturated outlined colour blobs, like patches optimised against person detectors.", _THYS),
    _spec("grating_multi", p_grating_multi, "optimisation_inspired", "High-frequency per-channel gratings.", _TSHIRT),
    _spec("bullseye", p_bullseye, "optimisation_inspired", "Concentric saturated rings.", _ADV_PATCH),
    # decoys
    _spec("stick_figures", p_stick_figures, "decoy", "Person-like cues; may INCREASE detection. Useful as a sanity check in the other direction.", ""),
]

REGISTRY = {s.name: s for s in SPECS}
CATEGORIES = list(dict.fromkeys(s.category for s in SPECS))


def generate(name, rng, size):
    out = REGISTRY[name].fn(rng, size)
    if out.shape[:2] != (size, size):
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out.astype(np.uint8))
