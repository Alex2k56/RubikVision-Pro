import json
from collections import Counter

import twophase.solver as sv
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from . import cv_utils

# Ordinea cerută de solver: U, R, F, D, L, B
FACE_ORDER = ["U", "R", "F", "D", "L", "B"]


def clean_moves(raw):
    """Scoate '(18f)' de la final și transformă R3 -> R', U1 -> U."""
    out = []
    for tok in str(raw).split():
        if tok.startswith("("):
            continue
        face, mod = tok[0], tok[1:]
        if mod == "3":
            mod = "'"
        elif mod == "1":
            mod = ""
        out.append(face + mod)
    return " ".join(out)


def index(request):
    return render(request, "solver/index.html")


@require_POST
def scan_face(request):
    data = json.loads(request.body)
    try:
        colors, feats = cv_utils.scan_face(data["image"])
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"colors": colors, "features": feats})


@require_POST
def solve(request):
    data = json.loads(request.body)
    faces = data.get("faces", [])

    # --- Validare ---
    if len(faces) != 6 or any(len(f["colors"]) != 9 for f in faces):
        return JsonResponse({"error": "Trebuie 6 fețe cu câte 9 stickere."}, status=400)

    centers = [f["colors"][4] for f in faces]
    if len(set(centers)) != 6:
        return JsonResponse(
            {"error": "Cele 6 centre trebuie să aibă 6 culori diferite. Verifică scanarea."},
            status=400,
        )

    counts = Counter(c for f in faces for c in f["colors"])
    bad = {c: n for c, n in counts.items() if n != 9}
    if bad:
        return JsonResponse(
            {"error": f"Fiecare culoare trebuie să apară de 9 ori. Acum: {dict(counts)}"},
            status=400,
        )

    # --- Construim șirul de 54 de caractere (culoare -> literă de față după centru) ---
    mapping = {centers[i]: FACE_ORDER[i] for i in range(6)}
    cube_string = "".join(mapping[c] for f in faces for c in f["colors"])

    try:
        raw = sv.solve(cube_string, 20, 5)  # max 20 mutări, timeout 5 secunde
    except Exception:
        raw = "Error"

    if not raw or str(raw).startswith("Error"):
        return JsonResponse(
            {"error": "Configurație imposibilă (scanare greșită sau orientare incorectă a unei fețe)."},
            status=400,
        )
    moves = clean_moves(raw)

    # --- Salvăm datele confirmate pentru antrenarea modelului ---
    if data.get("save_samples"):
        feats = [ft for f in faces for ft in f["features"]]
        labels = [c for f in faces for c in f["colors"]]
        cv_utils.append_samples(feats, labels)