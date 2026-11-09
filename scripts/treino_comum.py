"""
Funções compartilhadas pelas três fases do treino (fase1_optuna.py,
fase2_avaliacao.py, fase3_teste.py).

- folds agrupados por `origem` e estratificados pela faixa de área;
- treino do YOLO com os parâmetros fixos do experimento;
- medição da área (mm²) a partir das máscaras previstas e cálculo do MAPE.

A medição de área segue os notebooks `calculo_de_area` de 2026: maior máscara
de escala, soma das máscaras de fungo, área = px_fungo × 1600 / px_escala, com
conf 0,20, iou 0,60 e retina_masks. Imagens sem escala ou sem fungo detectados
recebem erro de 100% (ver docs/hiperparametros_optuna.md).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
TRAIN_AUG = ROOT / "data" / "train_aug"
TEST = ROOT / "data" / "test"
GABARITO = ROOT / "gabarito.csv"
AUGMENT_CSV = ROOT / "splits" / "augment.csv"
SPLIT_CSV = ROOT / "splits" / "split.csv"
RUNS = ROOT / "runs"

NOMES = {0: "escala", 1: "fungo"}
ID_ESCALA, ID_FUNGO = 0, 1
AREA_ESCALA_MM2 = 1600.0

# Inferência para a área: mesmos valores dos notebooks de 2026.
CONF = 0.20
IOU = 0.60
MIN_PX_ESCALA = 1000  # detecções de escala menores que isso são ruído

SEED = 42

# Fixos em todas as fases (docs/hiperparametros_optuna.md, "Parâmetros fixos e por quê").
# Os que distorcem a área já são 0 no padrão da Ultralytics, mas ficam explícitos.
FIXOS = dict(
    batch=16,
    seed=SEED,
    deterministic=True,
    pretrained=True,
    patience=100,
    single_cls=False,
    shear=0.0,
    perspective=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
)

# Baseline: exatamente a configuração de 2026 (tudo no padrão da Ultralytics).
PADRAO = {"modelo": "yolov8n-seg"}


# --------------------------------------------------------------------------- #
# Argumentos comuns
# --------------------------------------------------------------------------- #
def args_comuns(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--imgsz", type=int, required=True, choices=[640, 1024],
                    help="resolução de entrada (busca separada para cada uma)")
    ap.add_argument("--device", default=None, help="GPU, ex.: 0 ou 0,1 (padrão: automático)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="False", choices=["False", "ram", "disk"],
                    help="cache de imagens da Ultralytics (ram acelera se couber na memória)")
    ap.add_argument("--saida", type=Path, default=RUNS, help="pasta raiz dos resultados")
    # Só para teste rápido do código; não usar nos experimentos.
    ap.add_argument("--fracao", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--max-val", type=int, default=None, help=argparse.SUPPRESS)


def cache_arg(valor: str):
    return False if valor == "False" else valor


# --------------------------------------------------------------------------- #
# Dados e folds
# --------------------------------------------------------------------------- #
def carregar_gabarito() -> dict[str, float]:
    df = pd.read_csv(GABARITO)
    return dict(zip(df["identificador"].astype(str), df["Area(mm²)"].astype(float)))


def carregar_faixas() -> dict[str, int]:
    df = pd.read_csv(SPLIT_CSV)
    return dict(zip(df["arquivo"], df["faixa_area"]))


def carregar_manifesto_treino() -> pd.DataFrame:
    """Imagens de data/train_aug com a imagem original de origem de cada uma."""
    df = pd.read_csv(AUGMENT_CSV)
    df["imagem"] = [TRAIN_AUG / "images" / f"{a}.jpg" for a in df["arquivo"]]
    faltando = [str(p) for p in df["imagem"] if not p.exists()]
    if faltando:
        raise SystemExit(f"{len(faltando)} imagens de splits/augment.csv não estão em "
                         f"data/train_aug (ex.: {faltando[0]}). Rode scripts/augment.py.")
    return df


def gerar_folds(df: pd.DataFrame, n_folds: int, n_repeticoes: int = 1,
                seed: int = SEED) -> Iterator[tuple[str, list[Path], list[Path]]]:
    """
    K-Fold sobre as imagens ORIGINAIS, estratificado pela faixa de área.
    Cada cópia aumentada vai para o mesmo lado da sua origem (equivale ao
    GroupKFold por `origem`). A validação usa só as originais.

    Gera (nome, imagens_treino, imagens_validacao).
    """
    faixas = carregar_faixas()
    origens = np.array(sorted(df["origem"].unique()))
    estratos = np.array([faixas[o] for o in origens])
    for r in range(n_repeticoes):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for f, (_, idx_val) in enumerate(skf.split(origens, estratos)):
            val_origens = set(origens[idx_val])
            treino = df.loc[~df["origem"].isin(val_origens), "imagem"].tolist()
            val = df.loc[df["origem"].isin(val_origens) & ~df["aumentada"], "imagem"].tolist()
            nome = f"rep{r + 1}_fold{f + 1}" if n_repeticoes > 1 else f"fold{f + 1}"
            yield nome, treino, val


def escrever_dataset(pasta: Path, treino: list[Path], val: list[Path],
                     max_val: int | None = None) -> Path:
    """data.yaml que aponta para listas de imagens (labels em ../labels/)."""
    if max_val:  # só no teste rápido do código
        val = val[:max_val]
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "train.txt").write_text("\n".join(str(p) for p in treino) + "\n")
    (pasta / "val.txt").write_text("\n".join(str(p) for p in val) + "\n")
    cfg = {"path": str(ROOT), "train": str(pasta / "train.txt"),
           "val": str(pasta / "val.txt"), "names": NOMES}
    caminho = pasta / "data.yaml"
    caminho.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return caminho


# --------------------------------------------------------------------------- #
# Treino
# --------------------------------------------------------------------------- #
def treinar(params: dict, data_yaml: Path, epochs: int, imgsz: int, projeto: Path,
            nome: str, args: argparse.Namespace, plots: bool = False) -> dict:
    """
    Treina um modelo e devolve o caminho dos pesos finais e o mAP da máscara.

    A validação roda só na última época (val=False): a escolha é feita pelo
    MAPE da área, e a fase 3 não tem conjunto de validação. Por isso os pesos
    usados em todas as fases são os da última época.
    """
    from ultralytics import YOLO

    params = dict(params)
    modelo = params.pop("modelo")
    model = YOLO(f"{modelo}.pt")
    inicio = time.time()
    res = model.train(
        data=str(data_yaml), task="segment", epochs=epochs, imgsz=imgsz,
        project=str(projeto), name=nome, exist_ok=True, val=False, plots=plots,
        verbose=False, device=args.device, workers=args.workers,
        cache=cache_arg(args.cache), fraction=args.fracao,
        **{**FIXOS, **params},
    )
    pesos = Path(model.trainer.save_dir) / "weights" / "last.pt"
    return {
        "pesos": pesos,
        "segundos_treino": round(time.time() - inicio, 1),
        "mAP50_mask": float(res.seg.map50) if res is not None else float("nan"),
        "mAP50-95_mask": float(res.seg.map) if res is not None else float("nan"),
        "mAP50-95_box": float(res.box.map) if res is not None else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Área e MAPE
# --------------------------------------------------------------------------- #
def _areas_px(resultado) -> tuple[float, float, int]:
    """(px_escala, px_fungo, n_colonias) a partir das máscaras de um resultado."""
    px_escala, px_fungo, n_col = 0.0, 0.0, 0
    if resultado.masks is None:
        return px_escala, px_fungo, n_col
    for cls, poligono in zip(resultado.boxes.cls.tolist(), resultado.masks.xy):
        if len(poligono) < 3:
            continue
        area = float(cv2.contourArea(poligono.astype(np.float32)))
        if int(cls) == ID_ESCALA:
            px_escala = max(px_escala, area)
        elif int(cls) == ID_FUNGO:
            px_fungo += area
            n_col += 1
    return px_escala, px_fungo, n_col


def medir_areas(pesos: Path, imagens: list[Path], imgsz: int,
                args: argparse.Namespace) -> pd.DataFrame:
    """Área prevista × gabarito para cada imagem."""
    from ultralytics import YOLO

    gabarito = carregar_gabarito()
    faixas = carregar_faixas()
    if args.max_val:
        imagens = imagens[: args.max_val]
    model = YOLO(str(pesos))
    linhas = []
    resultados = model.predict([str(p) for p in imagens], stream=True, imgsz=imgsz,
                               conf=CONF, iou=IOU, retina_masks=True, verbose=False,
                               device=args.device)
    for img, r in zip(imagens, resultados):
        real = gabarito[img.stem]
        px_escala, px_fungo, n_col = _areas_px(r)
        if px_escala < MIN_PX_ESCALA:
            falha, prevista = "sem_escala", float("nan")
        elif px_fungo == 0:
            falha, prevista = "sem_fungo", 0.0
        else:
            falha, prevista = "", px_fungo * AREA_ESCALA_MM2 / px_escala
        erro = 100.0 if falha else abs(prevista - real) / real * 100
        linhas.append({
            "arquivo": img.stem, "faixa_area": faixas.get(img.stem),
            "real_mm2": real, "prevista_mm2": prevista, "erro_%": erro,
            "falha": falha, "px_escala": px_escala, "px_fungo": px_fungo,
            "n_colonias": n_col,
        })
    return pd.DataFrame(linhas)


def resumo_areas(df: pd.DataFrame) -> dict:
    """MAPE geral, por faixa de área e contagem de falhas."""
    por_faixa = df.groupby("faixa_area")["erro_%"].mean().round(3)
    return {
        "MAPE": float(df["erro_%"].mean()),
        "n_imagens": int(len(df)),
        "n_sem_escala": int((df["falha"] == "sem_escala").sum()),
        "n_sem_fungo": int((df["falha"] == "sem_fungo").sum()),
        **{f"MAPE_faixa{int(k)}": float(v) for k, v in por_faixa.items()},
    }


def salvar_json(caminho: Path, dados: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, indent=2, ensure_ascii=False) + "\n")


def carregar_config(nome: str, imgsz: int, saida: Path) -> tuple[str, dict]:
    """'padrao', 'otimizado' (resultado da fase 1) ou caminho de um .json."""
    if nome == "padrao":
        return "padrao", dict(PADRAO)
    if nome == "otimizado":
        caminho = saida / f"fase1_{imgsz}" / "melhores_parametros.json"
        if not caminho.exists():
            raise SystemExit(f"{caminho} não existe. Rode a fase 1 antes.")
        return "otimizado", json.loads(caminho.read_text())
    caminho = Path(nome)
    return caminho.stem, json.loads(caminho.read_text())


def anexar_csv(caminho: Path, linha: dict) -> None:
    """Acrescenta uma linha ao CSV (cria o arquivo se não existir)."""
    df = pd.DataFrame([linha])
    df.to_csv(caminho, mode="a", header=not caminho.exists(), index=False)


def ja_feitos(caminho: Path, coluna: str = "fold") -> set:
    if not caminho.exists():
        return set()
    return set(pd.read_csv(caminho)[coluna].astype(str))

