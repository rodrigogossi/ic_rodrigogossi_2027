"""
Fase 2 — compara a configuração otimizada com o baseline (valores padrão).

Os dois foram treinados nos mesmos 15 folds, então a comparação é pareada
fold a fold: teste de Wilcoxon (principal, não supõe normalidade) e teste t
pareado (complementar), sobre o MAPE da área e o mAP50-95 da máscara.

A configuração com menor MAPE médio é gravada em vencedora.json e é a que a
fase 3 usa por padrão.

Uso:
    python scripts/fase2_comparar.py --imgsz 640

Saídas em <saida>/fase2_<imgsz>/:
    comparacao.txt         resumo e testes estatísticos
    comparacao_folds.csv   métricas lado a lado por fold
    comparacao_faixas.csv  MAPE por faixa de área nas duas configurações
    vencedora.json         configuração usada pela fase 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

import treino_comum as tc

A, B = "padrao", "otimizado"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imgsz", type=int, required=True, choices=[640, 1024])
    ap.add_argument("--saida", type=Path, default=tc.RUNS)
    args = ap.parse_args()

    pasta = args.saida / f"fase2_{args.imgsz}"
    m = {c: pd.read_csv(pasta / c / "metricas_folds.csv").set_index("fold") for c in (A, B)}
    comuns = m[A].index.intersection(m[B].index)
    if len(comuns) < len(m[A]) or len(comuns) < len(m[B]):
        print(f"Aviso: só {len(comuns)} folds em comum entre as duas configurações.")

    linhas = [f"Fase 2 — {args.imgsz} px — {len(comuns)} folds pareados", ""]
    lado = pd.DataFrame(index=comuns)
    for met in ("MAPE", "mAP50-95_mask"):
        a, b = m[A].loc[comuns, met], m[B].loc[comuns, met]
        lado[f"{met}_{A}"], lado[f"{met}_{B}"] = a, b
        dif = b - a
        w = stats.wilcoxon(b, a) if len(comuns) > 1 else None
        t = stats.ttest_rel(b, a) if len(comuns) > 1 else None
        linhas += [
            f"{met}",
            f"  {A:<10} {a.mean():.4f} ± {a.std():.4f}",
            f"  {B:<10} {b.mean():.4f} ± {b.std():.4f}",
            f"  diferença ({B} − {A}): {dif.mean():+.4f} ± {dif.std():.4f}"
            f"   ({B} melhor em {int((dif < 0).sum() if met == 'MAPE' else (dif > 0).sum())}/{len(dif)} folds)",
        ]
        if w is not None:
            linhas += [f"  Wilcoxon pareado: W = {w.statistic:.1f}, p = {w.pvalue:.4g}",
                       f"  t pareado:        t = {t.statistic:.3f}, p = {t.pvalue:.4g}"]
        linhas.append("")
    lado.to_csv(pasta / "comparacao_folds.csv")

    faixas = {}
    for c in (A, B):
        areas = pd.read_csv(pasta / c / "areas.csv")
        areas = areas[areas["fold"].isin(comuns)]
        faixas[c] = areas.groupby("faixa_area")["erro_%"].mean()
    tab = pd.DataFrame({f"MAPE_{c}": faixas[c] for c in (A, B)}).round(3)
    tab.to_csv(pasta / "comparacao_faixas.csv")
    linhas += ["MAPE por faixa de área (decis da estratificação)", tab.to_string(), ""]

    venc = B if m[B].loc[comuns, "MAPE"].mean() < m[A].loc[comuns, "MAPE"].mean() else A
    cfg = json.loads((pasta / venc / "config.json").read_text())["params"]
    tc.salvar_json(pasta / "vencedora.json", cfg)
    linhas += [f"Vencedora (menor MAPE médio): {venc} -> {pasta / 'vencedora.json'}"]

    texto = "\n".join(linhas)
    (pasta / "comparacao.txt").write_text(texto + "\n")
    print(texto)


if __name__ == "__main__":
    main()
