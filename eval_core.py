# Adversarial Pattern Lab
# Roni Bandini, Oct 2026, GPL License
# Evaluation logic and metrics


import hashlib
import io
import time
import zipfile

import cv2
import numpy as np
import pandas as pd

from eim_detector import draw as _draw_model_view, selectTarget as _select_target
from pattern_library import REGISTRY, generate

TILE_MODES = ("repeat", "mirror", "stretch")


def sha(img):
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


def load_mask(mask_img, shape):
    if mask_img.ndim == 3 and mask_img.shape[2] == 4:
        mask = mask_img[:, :, 3]
    elif mask_img.ndim == 3:
        mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask = mask_img
    if mask.shape[:2] != shape[:2]:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return mask.astype(np.float32) / 255.0


def _cover(patch, h, w, tile_mode, scale):
    if tile_mode == "stretch":
        # "Fill" the box keeping the patch's aspect ratio (scale to cover, crop the centre) so it is never distorted.
        ph, pw = patch.shape[:2]
        k = max(h / ph, w / pw)
        big = cv2.resize(patch, (max(w, int(round(pw * k))), max(h, int(round(ph * k)))), interpolation=cv2.INTER_AREA)
        y0, x0 = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
        return big[y0:y0 + h, x0:x0 + w]
    side = max(8, int(patch.shape[0] * scale))
    p = cv2.resize(patch, (side, side), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if tile_mode == "mirror":
        p = np.vstack([np.hstack([p, p[:, ::-1]]), np.hstack([p[::-1], p[::-1, ::-1]])])
    reps = (int(np.ceil(h / p.shape[0])), int(np.ceil(w / p.shape[1])), 1)
    return np.tile(p, reps)[:h, :w]


def apply_pattern(person, patch, mask, shade=0.6, tile_mode="repeat", scale=1.0):
    h, w = person.shape[:2]
    if mask.sum() == 0:
        return person.copy()
    if tile_mode == "stretch":
        ys, xs = np.where(mask > 0.5)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tiled = np.zeros_like(person)
        tiled[y0:y1, x0:x1] = _cover(patch, y1 - y0, x1 - x0, "stretch", 1.0)
    else:
        tiled = _cover(patch, h, w, tile_mode, scale)

    gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY).astype(np.float32)
    region = mask > 0.5
    reference = gray[region].mean() if region.any() else 128.0
    luminance = np.clip(gray / max(reference, 1.0), 0.4, 1.4)[..., None]
    shaded = np.clip(tiled.astype(np.float32) * (1 - shade + shade * luminance), 0, 255)
    m = mask[..., None]
    return (person.astype(np.float32) * (1 - m) + shaded * m).astype(np.uint8)


def map_boxes(dets, view_shape, src_shape, resize_mode=None):

    vh, vw = view_shape[:2]
    H, W = src_shape[:2]
    if abs(vw / vh - W / H) < 0.01:
        resize_mode = "squash"  # same aspect ratio: every mode reduces to plain scaling
    out = []
    for d in dets:
        d = dict(d)
        if d.get("x") is not None:
            if resize_mode == "fit-shortest":
                s = min(W, H) / vw
                ox, oy = (W - min(W, H)) / 2, (H - min(W, H)) / 2
                d["x"], d["y"], d["w"], d["h"] = d["x"] * s + ox, d["y"] * s + oy, d["w"] * s, d["h"] * s
            elif resize_mode == "fit-longest":
                s = max(W, H) / vw
                ox, oy = (W - max(W, H)) / 2, (H - max(W, H)) / 2
                d["x"], d["y"], d["w"], d["h"] = d["x"] * s + ox, d["y"] * s + oy, d["w"] * s, d["h"] * s
            else:
                sx, sy = W / vw, H / vh
                d["x"], d["y"], d["w"], d["h"] = d["x"] * sx, d["y"] * sy, d["w"] * sx, d["h"] * sy
        out.append(d)
    return out


def draw_boxes(img, dets):
    out = img.copy()
    k = max(img.shape[:2]) / 600.0
    for d in dets:
        if d.get("x") is None:
            continue
        x, y, w, h = (int(round(d[c])) for c in ("x", "y", "w", "h"))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), max(1, int(round(2 * k))))
        cv2.putText(out, f"{d['label']} {d['conf'] * 100:.1f}%", (x, max(y - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * max(k, 0.8), (0, 255, 0), max(1, int(round(k))), cv2.LINE_AA)
    return out


def run_evaluation(detector, person, mask, pattern_names, variants=3, size=300, seed=0, shade=0.6,
                   tile_mode="repeat", scale=1.0, label=None, threshold=0.5, strategy="largest",
                   extra_patches=None, progress=None, select_target=None, draw=None, resize_mode=None, on_record=None):

    extra_patches = extra_patches or {}
    select_target = select_target or _select_target
    draw = draw or _draw_model_view
    jobs = [(n, v) for n in pattern_names for v in range(variants)]
    records = []

    def measure(img):
        dets, view, ms = detector.detect(img)
        sel = select_target(dets, label, strategy)
        conf = sel["conf"] if sel else 0.0
        return dets, view, ms, sel, conf

    def pictures(img, dets, view):

        mapped = map_boxes(dets, view.shape, img.shape, resize_mode)
        return draw_boxes(img, mapped), draw(view, dets), mapped

    dets, view, ms, sel, base_conf = measure(person)
    ann, mv, boxes = pictures(person, dets, view)
    records.append(dict(kind="baseline", pattern="baseline_clean", category="baseline", seed=-1,
                        conf=base_conf, delta=0.0, detected=base_conf >= threshold, n_results=len(dets),
                        selected_label=sel["label"] if sel else "", inference_ms=ms,
                        image=person.copy(), annotated=ann, model_view=mv, boxes=boxes, patch=None,
                        sha256=sha(person)))
    if on_record:
        on_record(records[0])

    seen = {records[0]["sha256"]: "baseline_clean"}
    for i, (name, v) in enumerate(jobs):
        s = seed + i
        rng = np.random.default_rng(s)
        if name in extra_patches:
            patch, cat = cv2.resize(extra_patches[name], (size, size)), "uploaded"
        else:
            patch, cat = generate(name, rng, size), REGISTRY[name].category
        comp = apply_pattern(person, patch, mask, shade, tile_mode, scale)
        dets, view, ms, sel, conf = measure(comp)
        digest = sha(comp)
        ann, mv, boxes = pictures(comp, dets, view)
        records.append(dict(kind="test", pattern=name, category=cat, seed=s, conf=conf, delta=conf - base_conf,
                            detected=conf >= threshold, n_results=len(dets),
                            selected_label=sel["label"] if sel else "", inference_ms=ms, image=comp,
                            annotated=ann, model_view=mv, boxes=boxes, patch=patch, sha256=digest,
                            duplicate_of=seen.get(digest, "")))
        seen.setdefault(digest, f"{name}_s{s}")
        if on_record:
            on_record(records[-1])
        if progress:
            progress((i + 1) / max(len(jobs), 1), f"{name} (seed {s})")
    return records


def export_successful(records, hires=1200):

    wins = sorted((r for r in records if r["kind"] == "test" and not r["detected"]), key=lambda r: r["conf"])
    buf, rows = io.BytesIO(), []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def put(path, img):
            ok, png = cv2.imencode(".png", img)
            if ok:
                z.writestr(path, png.tobytes())
        for r in wins:
            stem = f'{r["pattern"]}_s{r["seed"]}'
            put(f"composites/{stem}.png", r["image"])
            if r.get("patch") is not None:
                put(f"patches/{stem}_tested.png", r["patch"])
            if r["pattern"] in REGISTRY:
                put(f"patches/{stem}_{hires}.png", generate(r["pattern"], np.random.default_rng(r["seed"]), hires))
            rows.append(dict(pattern=r["pattern"], category=r["category"], seed=r["seed"],
                             conf=round(float(r["conf"]), 6), delta=round(float(r["delta"]), 6)))
        z.writestr("successful.csv", pd.DataFrame(rows, columns=["pattern", "category", "seed", "conf", "delta"]).to_csv(index=False))
    return buf.getvalue(), len(wins)


def to_frame(records):
    cols = ["pattern", "category", "seed", "conf", "delta", "detected", "n_results", "selected_label",
            "inference_ms", "sha256"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in records])


def summarize(df):
    t = df[df["category"] != "baseline"].copy()
    if t.empty:
        return t, t
    agg = dict(runs=("conf", "size"), detection_rate=("detected", "mean"), mean_conf=("conf", "mean"),
               min_conf=("conf", "min"), mean_delta=("delta", "mean"))
    return (t.groupby("pattern").agg(**agg).sort_values("detection_rate").reset_index(),
            t.groupby("category").agg(**agg).sort_values("detection_rate").reset_index())
