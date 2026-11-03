"""
Desenha os polígonos YOLO sobre as imagens para conferência visual.

Gera um mosaico com a imagem original e suas cópias aumentadas lado a lado,
para checar se os labels acompanharam a transformação.

Uso:
    python scripts/visualizar_labels.py                 # 6 originais aleatórias
    python scripts/visualizar_labels.py --k 10 --dir data/train_aug
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "checks"
CORES = {0: (0, 0, 255), 1: (0, 255, 0)}  # BGR: escala=vermelho, fungo=verde
NOMES = {0: "escala", 1: "fungo"}
THUMB_H = 600


def draw(img_path: Path, lbl_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    overlay = img.copy()
    for line in lbl_path.read_text().split("\n"):
        vals = line.split()
        if not vals:
            continue
        cls = int(vals[0])
        pts = (np.array(vals[1:], dtype=np.float64).reshape(-1, 2) * (w, h)).astype(np.int32)
        cv2.fillPoly(overlay, [pts], CORES[cls])
        cv2.polylines(img, [pts], True, CORES[cls], max(2, w // 400))
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    scale = THUMB_H / h
    img = cv2.resize(img, (int(w * scale), THUMB_H))
    cv2.putText(img, img_path.stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/train_aug")
    ap.add_argument("--k", type=int, default=6, help="quantas originais sortear")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = ROOT / args.dir
    imgs = sorted((base / "images").glob("*.jpg"))
    originais = [p for p in imgs if "_aug" not in p.stem]
    random.Random(args.seed).shuffle(originais)
    OUT.mkdir(exist_ok=True)

    for orig in originais[: args.k]:
        grupo = [orig] + sorted(p for p in imgs if p.stem.startswith(orig.stem + "_aug"))
        tiles = [draw(p, base / "labels" / f"{p.stem}.txt") for p in grupo]
        mosaic = np.hstack([cv2.copyMakeBorder(t, 0, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles])
        dst = OUT / f"{orig.stem}.jpg"
        cv2.imwrite(str(dst), mosaic)
        print(dst.relative_to(ROOT))


if __name__ == "__main__":
    main()
