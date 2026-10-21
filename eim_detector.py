# Adversarial Pattern Lab
# Roni Bandini, Buenos Aires - Argentina - 10/2026 
# GPL License
# Evaluates mask against Edge Impulse ML model


import os
import sys
import time

import cv2
import numpy as np

ImageImpulseRunner = None
_IMPORT_ERROR = None
try:
    from edge_impulse_linux.image import ImageImpulseRunner
except ImportError as exc:
    _IMPORT_ERROR = exc


class EimDetector:
    def __init__(self, modelPath):
        if ImageImpulseRunner is None:
            raise SystemExit(
                f"Falta edge_impulse_linux: {_IMPORT_ERROR}\n"
                "Instalar: pip install edge_impulse_linux"
            )
        if not sys.platform.startswith("linux"):
            raise SystemExit("Solo para Linux, sorry :(")
        path = os.path.abspath(modelPath)
        if not os.path.isfile(path):
            raise SystemExit(f"No se encuentra el modelo: {path}")
        os.chmod(path, os.stat(path).st_mode | 0o111)
        self.path = path
        self.runner = ImageImpulseRunner(path)
        self.info = self.runner.init()
        params = self.info.get("model_parameters", {})
        self.labels = params.get("labels", [])
        project = self.info.get("project", {})
        print(f"Loaded model: {project.get('owner', '?')} / {project.get('name', '?')} labels={self.labels}")

    def detect(self, imgBgr):
        # Ejecuta y obtiene inferencias
        rgb = cv2.cvtColor(imgBgr, cv2.COLOR_BGR2RGB)
        try:
            features, cropped = self.runner.get_features_from_image_auto_studio_settings(rgb)
        except AttributeError:
            features, cropped = self.runner.get_features_from_image(rgb)
        start = time.perf_counter()
        response = self.runner.classify(features)          # Inferencia acá
        elapsedMs = (time.perf_counter() - start) * 1000
        result = response.get("result", {})
        detections = []
        if "bounding_boxes" in result:                      # detección de objetos con bounding boxes
            for box in result["bounding_boxes"]:
                detections.append({
                    "label": box["label"], "conf": float(box["value"]),
                    "x": int(box["x"]), "y": int(box["y"]),
                    "w": int(box["width"]), "h": int(box["height"]),
                })
        elif "classification" in result:                    # Clasificación
            for label, value in result["classification"].items():
                detections.append({"label": label, "conf": float(value),
                                   "x": None, "y": None, "w": None, "h": None})
        cropped = np.ascontiguousarray(cropped)
        if cropped.ndim == 2:
            view = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        else:
            view = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        return detections, view, elapsedMs

    def close(self):
        try:
            self.runner.stop()
        except Exception:
            pass


def draw(view, detections):
    # Dibuja los recuadros
    annotated = view.copy()
    for item in detections:
        if item["x"] is None:
            continue
        x, y, w, h = item["x"], item["y"], item["w"], item["h"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"{item['label']} {item['conf'] * 100:.1f}%",
                    (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return annotated


def selectTarget(detections, label, strategy):
    candidates = [item for item in detections if label is None or item["label"] == label]
    if not candidates:
        return None
    boxes = [item for item in candidates if item["x"] is not None]
    if not boxes:
        return max(candidates, key=lambda item: item["conf"])
    if strategy == "largest":
        return max(boxes, key=lambda item: (item["w"] * item["h"], item["conf"]))
    return max(candidates, key=lambda item: item["conf"])
