import base64
import csv
from pathlib import Path

import cv2
import joblib
import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "color_model.joblib"
DATASET_PATH = BASE / "dataset.csv"

_model = None


def _load_model():
    """Încarcă modelul Random Forest dacă există (altfel folosim reguli HSV)."""
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model


def decode_image(data_url):
    b64 = data_url.split(",", 1)[-1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagine invalidă.")
    return img


def extract_features(img):
    """
    Imaginea primită este deja decupată pătrat (doar fața cubului).
    O împărțim în 3x3 și, pentru fiecare celulă, luăm mediana culorii
    din zona centrală (ca să evităm marginile negre dintre stickere).
    Features: L, a, b (LAB) + H, S, V (HSV).
    """
    h, w = img.shape[:2]
    side = min(h, w)
    img = img[:side, :side]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cell = side // 3
    margin = int(cell * 0.3)

    feats = []
    for r in range(3):
        for c in range(3):
            ys, ye = r * cell + margin, (r + 1) * cell - margin
            xs, xe = c * cell + margin, (c + 1) * cell - margin
            l_med = np.median(lab[ys:ye, xs:xe].reshape(-1, 3), axis=0)
            h_med = np.median(hsv[ys:ye, xs:xe].reshape(-1, 3), axis=0)
            feats.append([float(x) for x in (*l_med, *h_med)])
    return feats


def rule_based(f):
    """Fallback simplu pe HSV, dacă nu ai antrenat încă modelul. Ajustează pragurile."""
    h, s, v = f[3], f[4], f[5]
    if s < 70 and v > 110:
        return "W"
    if h < 7 or h >= 168:
        return "R"
    if h < 20:
        return "O"
    if h < 38:
        return "Y"
    if h < 90:
        return "G"
    return "B"


def scan_face(data_url):
    img = decode_image(data_url)
    feats = extract_features(img)
    model = _load_model()
    if model is not None:
        colors = [str(p) for p in model.predict(np.array(feats))]
    else:
        colors = [rule_based(f) for f in feats]
    return colors, feats


def append_samples(feats, labels):
    """Salvează culorile confirmate de utilizator ca date de antrenare."""
    is_new = not DATASET_PATH.exists()
    with open(DATASET_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["L", "a", "b", "H", "S", "V", "label"])
        for ft, lb in zip(feats, labels):
            w.writerow(list(ft) + [lb])