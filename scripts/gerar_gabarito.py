"""
Gera o gabarito de área (mm²) de cada imagem a partir dos polígonos do Label Studio.

Adaptado de gerar_gabarito.py do repo de 2026 (ic_rodrigogossi_yollov8). O
cálculo é idêntico, então o resultado reproduz exatamente o gabarito.csv de 2026.

ATENÇÃO: este gabarito NÃO é a medição manual do ImageJ. É a área das próprias
anotações usadas no treino, convertida para mm² pela escala física:

    área_fungo_mm² = área_fungo_px × 1600 / área_escala_px

onde 1600 mm² é a área do quadrado de referência (4 cm × 4 cm).

Regras (iguais às de 2026):
    - escala: se houver mais de um polígono, usa o MAIOR
    - fungo: SOMA de todos os polígonos de fungo da imagem
    - coordenadas desnormalizadas e truncadas para pixel inteiro (int32)
    - área arredondada em 2 casas decimais

Saída:
    gabarito.csv  (colunas: identificador, Area(mm²))

Uso:
    python scripts/gerar_gabarito.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SAIDA = ROOT / "gabarito.csv"

AREA_ESCALA_MM2 = (4.0 * 10) ** 2  # quadrado de 4 cm × 4 cm = 1600 mm²
CLS_ESCALA, CLS_FUNGO = 0, 1


def area_mm2(img_path: Path, lbl_path: Path) -> float:
    h, w = cv2.imread(str(img_path)).shape[:2]
    area_escala = area_fungo = 0.0
    for line in lbl_path.read_text().split("\n"):
        vals = line.split()
        if not vals:
            continue
        cls = int(vals[0])
        pts = np.array(vals[1:], dtype=np.float32).reshape(-1, 2) * np.array([w, h])
        area = cv2.contourArea(pts.astype(np.int32))
        if cls == CLS_ESCALA:
            area_escala = max(area_escala, area)
        elif cls == CLS_FUNGO:
            area_fungo += area
    if area_escala == 0:
        print(f"AVISO: {img_path.stem} sem escala anotada, área = 0")
        return 0.0
    return round(area_fungo * AREA_ESCALA_MM2 / area_escala, 2)


def main() -> None:
    rows = []
    for lbl in sorted((RAW / "labels").glob("*.txt")):
        img = RAW / "images" / f"{lbl.stem}.jpg"
        if not img.exists():
            sys.exit(f"Imagem não encontrada para {lbl.name}")
        rows.append({"identificador": lbl.stem, "Area(mm²)": area_mm2(img, lbl)})
    df = pd.DataFrame(rows)
    df.to_csv(SAIDA, index=False)
    print(f"{len(df)} imagens -> {SAIDA.relative_to(ROOT)}")
    print(df["Area(mm²)"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
