"""
Desenha os polígonos YOLO sobre as imagens para conferência visual.

Gera um mosaico com a imagem original e suas cópias aumentadas lado a lado,
para checar se os labels acompanharam a transformação.

Uso:
    python scripts/visualizar_labels.py                 # 6 originais aleatórias
    python scripts/visualizar_labels.py --k 10 --dir data/train_aug

Figura de demonstração (legendas embaixo, sem nome de arquivo):
    python scripts/visualizar_labels.py --imagem d5_t4_60uL_r3 --demo \
        --saida docs/img/exemplo_augmentation.jpg            # com os polígonos
    python scripts/visualizar_labels.py --imagem d5_t4_60uL_r3 --demo --sem-labels \
        --saida docs/img/exemplo_augmentation_sem_labels.jpg # só as imagens
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
LEGENDA_H = 44


def draw(img_path: Path, lbl_path: Path, labels: bool = True, nome: bool = True) -> np.ndarray:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    if labels:
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
    img = cv2.resize(img, (int(w * scale), THUMB_H), interpolation=cv2.INTER_AREA)
    if nome:
        cv2.putText(img, img_path.stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def legenda(tile: np.ndarray, texto: str) -> np.ndarray:
    """Acrescenta uma faixa branca com o texto centralizado embaixo da imagem."""
    faixa = np.full((LEGENDA_H, tile.shape[1], 3), 255, np.uint8)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(faixa, texto, ((tile.shape[1] - tw) // 2, (LEGENDA_H + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    return np.vstack([tile, faixa])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/train_aug")
    ap.add_argument("--k", type=int, default=6, help="quantas originais sortear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imagem", help="nome (sem extensão) de uma imagem original específica")
    ap.add_argument("--demo", action="store_true", help="legendas embaixo, sem nome de arquivo")
    ap.add_argument("--sem-labels", action="store_true", help="não desenha os polígonos")
    ap.add_argument("--saida", help="arquivo de saída (só com --imagem)")
    args = ap.parse_args()

    base = ROOT / args.dir
    imgs = sorted((base / "images").glob("*.jpg"))
    originais = [p for p in imgs if "_aug" not in p.stem]
    if args.imagem:
        originais = [p for p in originais if p.stem == args.imagem]
        if not originais:
            ap.error(f"{args.imagem} não está em {args.dir} (pode estar no teste)")
    else:
        random.Random(args.seed).shuffle(originais)
    OUT.mkdir(exist_ok=True)

    for orig in originais[: args.k]:
        grupo = [orig] + sorted(p for p in imgs if p.stem.startswith(orig.stem + "_aug"))
        tiles = [draw(p, base / "labels" / f"{p.stem}.txt", labels=not args.sem_labels, nome=not args.demo)
                 for p in grupo]
        if args.demo:
            tiles = [legenda(t, "Original" if i == 0 else f"Aumentada {i}") for i, t in enumerate(tiles)]
        mosaic = np.hstack([cv2.copyMakeBorder(t, 0, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles])
        dst = ROOT / args.saida if args.saida and args.imagem else OUT / f"{orig.stem}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(dst.relative_to(ROOT))


if __name__ == "__main__":
    main()
