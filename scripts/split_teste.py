"""
Separa 20% do dataset bruto como conjunto de TESTE isolado.

A separação é ESTRATIFICADA por imagem, usando apenas a ÁREA DA COLÔNIA:
as imagens são divididas em faixas de área (decis da área do gabarito.csv,
calculada dos polígonos do Label Studio com a escala de 1600 mm²) e 20% de
cada faixa vai para o teste. Assim o teste cobre todos os
tamanhos de colônia na mesma proporção do dataset completo.

A representatividade é verificada estatisticamente (KS para área, qui-quadrado
para dia/tratamento/volume) e salva em splits/representatividade.{txt,png}.

Saídas:
    data/test/{images,labels}        -> teste congelado (NUNCA usar no treino)
    data/train_base/{images,labels}  -> 80% restantes (entrada do augment.py)
    splits/split.csv                 -> manifesto (arquivo, faixa de área, conjunto, sha256)

Uso:
    python scripts/split_teste.py            # cria a separação
    python scripts/split_teste.py --verify   # confere se o teste não foi alterado
    python scripts/split_teste.py --report   # refaz só o relatório de representatividade
"""
import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GABARITO = ROOT / "gabarito.csv"
TEST = ROOT / "data" / "test"
TRAIN = ROOT / "data" / "train_base"
SPLITS = ROOT / "splits"
MANIFEST = SPLITS / "split.csv"

SEED = 42
TEST_FRACTION = 0.2
N_FAIXAS_AREA = 10

# d1_t2_100uL_r3B -> dia=1, trat=t2, vol=100uL, rep=r3 (o "B" é refoto da mesma placa)
NAME_RE = re.compile(r"^d(?P<dia>\d+)_(?P<trat>t\d+)_(?P<vol>\d+uL)_(?P<rep>r\d+)B?$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_table() -> pd.DataFrame:
    area = pd.read_csv(GABARITO).set_index("identificador")["Area(mm²)"]
    rows = []
    for img in sorted((RAW / "images").glob("*.jpg")):
        m = NAME_RE.match(img.stem)
        if not m:
            sys.exit(f"Nome fora do padrão: {img.name}")
        if not (RAW / "labels" / f"{img.stem}.txt").exists():
            sys.exit(f"Imagem sem label: {img.name}")
        if img.stem not in area.index:
            sys.exit(f"Imagem sem área no gabarito: {img.name}")
        rows.append({
            "arquivo": img.stem,
            "dia": f"d{m['dia']}",
            "tratamento": m["trat"],
            "volume": m["vol"],
            "placa": f"{m['trat']}_{m['vol']}_{m['rep']}",
            "area_mm2": area[img.stem],
        })
    df = pd.DataFrame(rows)
    df["faixa_area"] = pd.qcut(df["area_mm2"], N_FAIXAS_AREA, labels=False)
    return df


def split(df: pd.DataFrame) -> pd.DataFrame:
    _, test_idx = train_test_split(
        df.index, test_size=TEST_FRACTION, stratify=df["faixa_area"], random_state=SEED,
    )
    df = df.copy()
    df["conjunto"] = "train"
    df.loc[test_idx, "conjunto"] = "test"
    return df


def copy_split(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        dst = TEST if row["conjunto"] == "test" else TRAIN
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            (dst / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy2(RAW / sub / f"{row['arquivo']}{ext}", dst / sub / f"{row['arquivo']}{ext}")


def add_hashes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sha256_img"] = [sha256(RAW / "images" / f"{a}.jpg") for a in df["arquivo"]]
    df["sha256_lbl"] = [sha256(RAW / "labels" / f"{a}.txt") for a in df["arquivo"]]
    return df


def report(df: pd.DataFrame) -> None:
    """Compara teste x dataset completo e salva texto + figura."""
    test = df[df["conjunto"] == "test"]
    train = df[df["conjunto"] == "train"]
    linhas = [
        f"Total: {len(df)} | Teste: {len(test)} ({len(test) / len(df):.1%}) | Treino: {len(train)}",
        "",
        "Área da colônia (mm²) — quantis",
    ]
    q = [0, .1, .25, .5, .75, .9, 1]
    tab = pd.DataFrame({
        "completo": df["area_mm2"].quantile(q),
        "treino": train["area_mm2"].quantile(q),
        "teste": test["area_mm2"].quantile(q),
    }).round(1)
    tab.index = ["mín", "p10", "p25", "mediana", "p75", "p90", "máx"]
    linhas.append(tab.to_string())
    ks = stats.ks_2samp(train["area_mm2"], test["area_mm2"])
    linhas += ["", f"Kolmogorov-Smirnov treino x teste (área): D={ks.statistic:.3f}  p={ks.pvalue:.3f}",
               "  (p > 0,05 => não há evidência de que as distribuições de área diferem)", ""]

    for col in ("faixa_area", "tratamento", "volume", "dia"):
        ct = pd.crosstab(df[col], df["conjunto"])
        chi = stats.chi2_contingency(ct)
        pct = (ct["test"] / ct.sum(axis=1) * 100).round(1)
        linhas.append(f"{col}: qui-quadrado p={chi.pvalue:.3f} | % no teste por categoria: "
                      + ", ".join(f"{k}={v}" for k, v in pct.items()))
    texto = "\n".join(linhas)
    (SPLITS / "representatividade.txt").write_text(texto + "\n")
    print(texto)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    bins = np.histogram_bin_edges(df["area_mm2"], 25)
    axes[0].hist(train["area_mm2"], bins, density=True, alpha=.5, label="treino")
    axes[0].hist(test["area_mm2"], bins, density=True, alpha=.5, label="teste")
    axes[0].set(title=f"Área (KS p={ks.pvalue:.2f})", xlabel="mm²")
    axes[0].legend()
    for ax, col in zip(axes[1:], ("tratamento", "volume", "dia")):
        prop = pd.crosstab(df[col], df["conjunto"], normalize="columns")[["train", "test"]]
        prop.columns = ["treino", "teste"]
        prop.plot.bar(ax=ax, rot=0)
        ax.set(title=f"Proporção por {col}", xlabel="", ylabel="fração")
    fig.tight_layout()
    fig.savefig(SPLITS / "representatividade.png", dpi=120)


def verify() -> None:
    df = pd.read_csv(MANIFEST)
    test = df[df["conjunto"] == "test"]
    erros = 0
    for _, row in test.iterrows():
        for sub, ext, col in (("images", ".jpg", "sha256_img"), ("labels", ".txt", "sha256_lbl")):
            p = TEST / sub / f"{row['arquivo']}{ext}"
            if not p.exists() or sha256(p) != row[col]:
                print(f"ALTERADO/AUSENTE: {p.relative_to(ROOT)}")
                erros += 1
    extras = {p.stem for p in (TEST / "images").glob("*")} - set(test["arquivo"])
    for e in sorted(extras):
        print(f"EXTRA no teste: {e}")
        erros += 1
    no_treino = {p.stem for p in (TRAIN / "images").glob("*")} & set(test["arquivo"])
    for e in sorted(no_treino):
        print(f"Imagem de teste também no treino: {e}")
        erros += 1
    if erros:
        sys.exit(f"{erros} problema(s) encontrados.")
    print(f"OK: {len(test)} imagens de teste íntegras e ausentes do treino.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="confere integridade do teste")
    ap.add_argument("--report", action="store_true", help="refaz só o relatório de representatividade")
    ap.add_argument("--force", action="store_true", help="refaz a separação mesmo se já existir")
    args = ap.parse_args()

    if args.verify:
        verify()
        return
    if args.report:
        report(pd.read_csv(MANIFEST))
        return

    if MANIFEST.exists() and not args.force:
        sys.exit(f"{MANIFEST.relative_to(ROOT)} já existe. O teste é congelado; use --force só se tiver certeza.")
    for d in (TEST, TRAIN):
        if d.exists():
            shutil.rmtree(d)

    df = add_hashes(split(build_table()))
    copy_split(df)
    SPLITS.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    report(df)


if __name__ == "__main__":
    main()
