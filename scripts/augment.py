"""
Data augmentation offline do conjunto de TREINO (data/train_base).

Cada polígono YOLO é rasterizado como máscara, a MESMA transformação é aplicada
à imagem e às máscaras, e as máscaras transformadas voltam a ser polígonos YOLO.
Ou seja: os labels das imagens aumentadas são gerados automaticamente.

Restrições importantes para medição de área (fungo/escala):
    - só transformações geométricas que preservam a razão de áreas
      (flip, rotação, escala UNIFORME, translação). Sem shear, perspectiva
      ou distorção elástica.
    - nenhuma instância pode ser cortada pela borda; se for, a amostra é
      sorteada de novo.

O conjunto de teste (data/test) NUNCA é lido por este script.

Saídas:
    data/train_aug/{images,labels}  -> originais + N cópias aumentadas de cada
    splits/augment.csv              -> origem, placa e checagem de razão de área

Uso:
    python scripts/augment.py               # 4 cópias por imagem
    python scripts/augment.py --n 3 --max-side 1600
"""
import argparse
import re
import shutil
import zlib
from multiprocessing import Pool
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "train_base"
DST = ROOT / "data" / "train_aug"
MANIFEST = ROOT / "splits" / "augment.csv"

SEED = 42
MAX_TRIES = 20
CLS_ESCALA, CLS_FUNGO = 0, 1
PLACA_RE = re.compile(r"^d\d+_(t\d+_\d+uL_r\d+)B?$")


def build_pipeline(max_side: int, seed: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=max_side, p=1.0),
        # --- geométricas (preservam razão de áreas) ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.85, 1.15), keep_ratio=True,
            translate_percent=(-0.05, 0.05),
            rotate=(-180, 180),
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
            p=0.7,
        ),
        # --- fotométricas (iluminação / câmera) ---
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.CLAHE(clip_limit=(1, 3)),
        ], p=0.8),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.06)),
            A.ISONoise(),
        ], p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.2),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), shadow_intensity_range=(0.2, 0.4), p=0.15),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
    ], seed=seed)


def read_label(path: Path, w: int, h: int):
    """Retorna [(classe, máscara uint8, área do polígono em px)] a partir de um label YOLO-seg."""
    instances = []
    for line in path.read_text().split("\n"):
        vals = line.split()
        if not vals:
            continue
        cls = int(vals[0])
        pts = np.array(vals[1:], dtype=np.float64).reshape(-1, 2) * (w, h)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        instances.append((cls, mask, cv2.contourArea(pts.astype(np.float32))))
    return instances


def mask_to_polygon(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    return c if len(c) >= 3 else None


def touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def area_ratio(classes, areas):
    """Razão área_fungo / área_escala (o que o pipeline de medição usa)."""
    esc = sum(a for c, a in zip(classes, areas) if c == CLS_ESCALA)
    fun = sum(a for c, a in zip(classes, areas) if c == CLS_FUNGO)
    return fun / esc if esc else np.nan


def write_label(path: Path, classes, polys, w: int, h: int) -> None:
    lines = []
    for cls, poly in zip(classes, polys):
        norm = (poly / (w, h)).clip(0, 1).ravel()
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm))
    path.write_text("\n".join(lines) + "\n")


def process(args):
    img_path, n_aug, max_side = args
    stem = img_path.stem
    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    instances = read_label(SRC / "labels" / f"{stem}.txt", w, h)
    classes = [c for c, _, _ in instances]
    masks = [m for _, m, _ in instances]
    orig_ratio = area_ratio(classes, [a for _, _, a in instances])
    placa = PLACA_RE.match(stem).group(1)

    shutil.copy2(img_path, DST / "images" / img_path.name)
    shutil.copy2(SRC / "labels" / f"{stem}.txt", DST / "labels" / f"{stem}.txt")
    rows = [{"arquivo": stem, "origem": stem, "placa": placa, "aumentada": False,
             "tentativas": 0, "razao_original": orig_ratio, "razao_nova": orig_ratio}]

    pipeline = build_pipeline(max_side, seed=SEED + zlib.crc32(stem.encode()))
    for k in range(1, n_aug + 1):
        for tries in range(1, MAX_TRIES + 1):
            out = pipeline(image=image, masks=masks)
            new_masks = out["masks"]
            if any(touches_border(m) or not m.any() for m in new_masks):
                continue
            polys = [mask_to_polygon(np.ascontiguousarray(m)) for m in new_masks]
            if any(p is None for p in polys):
                continue
            break
        else:
            print(f"AVISO: {stem} aug{k} descartada após {MAX_TRIES} tentativas")
            continue

        oh, ow = out["image"].shape[:2]
        name = f"{stem}_aug{k}"
        cv2.imwrite(str(DST / "images" / f"{name}.jpg"),
                    cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_label(DST / "labels" / f"{name}.txt", classes, polys, ow, oh)
        rows.append({"arquivo": name, "origem": stem, "placa": placa, "aumentada": True,
                     "tentativas": tries, "razao_original": orig_ratio,
                     "razao_nova": area_ratio(classes, [cv2.contourArea(p.astype(np.float32)) for p in polys])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="cópias aumentadas por imagem")
    ap.add_argument("--max-side", type=int, default=2048, help="lado maior das imagens aumentadas (px)")
    ap.add_argument("--limit", type=int, default=None, help="processa só as N primeiras imagens (teste rápido)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    if DST.exists():
        shutil.rmtree(DST)
    (DST / "images").mkdir(parents=True)
    (DST / "labels").mkdir(parents=True)

    imgs = sorted((SRC / "images").glob("*.jpg"))[: args.limit]
    jobs = [(p, args.n, args.max_side) for p in imgs]
    rows = []
    with Pool(args.workers) as pool:
        for r in tqdm(pool.imap_unordered(process, jobs), total=len(jobs), desc="augment"):
            rows += r

    df = pd.DataFrame(rows).sort_values("arquivo")
    df["erro_razao_%"] = 100 * (df["razao_nova"] - df["razao_original"]).abs() / df["razao_original"]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)

    aug = df[df["aumentada"]]
    print(f"\n{len(imgs)} originais + {len(aug)} aumentadas = {len(df)} imagens em {DST.relative_to(ROOT)}")
    print(f"Erro na razão fungo/escala após transformação: "
          f"mediana {aug['erro_razao_%'].median():.2f}% | p95 {aug['erro_razao_%'].quantile(.95):.2f}% "
          f"| máx {aug['erro_razao_%'].max():.2f}%")


if __name__ == "__main__":
    main()
