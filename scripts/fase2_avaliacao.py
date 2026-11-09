"""
Fase 2 — avaliação robusta: Repeated K-Fold 3×5 com 100 épocas, como em 2026.

Roda uma configuração por vez, sempre com os mesmos folds, para permitir o
teste pareado em fase2_comparar.py:
    --config padrao     baseline: valores padrão da Ultralytics (YOLOv8n-seg, como em 2026)
    --config otimizado  melhores_parametros.json da fase 1
    --config arq.json   qualquer outra configuração

Se a execução cair, rode o mesmo comando: os folds já concluídos são pulados.

Uso:
    python scripts/fase2_avaliacao.py --imgsz 640 --config padrao
    python scripts/fase2_avaliacao.py --imgsz 640 --config otimizado

    # medir o tempo real antes da busca: 1 fold, poucas épocas
    python scripts/fase2_avaliacao.py --imgsz 640 --config padrao --max-folds 1 --epochs 3 --saida runs/tempo

Saídas em <saida>/fase2_<imgsz>/<config>/:
    metricas_folds.csv   uma linha por fold (MAPE, mAP, falhas, tempo)
    areas.csv            área prevista × gabarito de cada imagem de validação
    config.json          parâmetros usados
"""
from __future__ import annotations

import argparse
import shutil

import pandas as pd

import treino_comum as tc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.args_comuns(ap)
    ap.add_argument("--config", required=True, help="padrao, otimizado ou caminho de um .json")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeticoes", type=int, default=3)
    ap.add_argument("--max-folds", type=int, default=None, help="para depois de N folds (medição de tempo)")
    ap.add_argument("--apagar-pesos", action="store_true",
                    help="apaga os pesos de cada fold depois de medir a área")
    args = ap.parse_args()

    nome_cfg, params = tc.carregar_config(args.config, args.imgsz, args.saida)
    pasta = args.saida / f"fase2_{args.imgsz}" / nome_cfg
    pasta.mkdir(parents=True, exist_ok=True)
    tc.salvar_json(pasta / "config.json", {"params": params, "fixos": tc.FIXOS,
                                           "epochs": args.epochs, "imgsz": args.imgsz})
    csv_metricas, csv_areas = pasta / "metricas_folds.csv", pasta / "areas.csv"
    feitos = tc.ja_feitos(csv_metricas)

    df = tc.carregar_manifesto_treino()
    folds = list(tc.gerar_folds(df, args.folds, args.repeticoes))
    if args.max_folds:
        folds = folds[: args.max_folds]

    for k, (nome, treino, val) in enumerate(folds, start=1):
        if nome in feitos:
            print(f"[{k}/{len(folds)}] {nome}: já concluído, pulando.")
            continue
        print(f"\n[{k}/{len(folds)}] {nome}  {nome_cfg}  {args.imgsz}px  "
              f"({len(treino)} imagens de treino, {len(val)} de validação)")
        data_yaml = tc.escrever_dataset(pasta / nome / "dados", treino, val, args.max_val)
        r = tc.treinar(params, data_yaml, args.epochs, args.imgsz, pasta, nome, args, plots=True)
        areas = tc.medir_areas(r["pesos"], val, args.imgsz, args)
        areas.insert(0, "fold", nome)
        if csv_areas.exists():  # descarta linhas de uma execução interrompida deste fold
            antigas = pd.read_csv(csv_areas)
            areas = pd.concat([antigas[antigas["fold"] != nome], areas])
        areas.to_csv(csv_areas, index=False)
        areas = areas[areas["fold"] == nome]
        resumo = tc.resumo_areas(areas)
        tc.anexar_csv(csv_metricas, {
            "fold": nome, "repeticao": nome.split("_")[0],
            "MAPE": round(resumo["MAPE"], 4),
            "mAP50_mask": round(r["mAP50_mask"], 4),
            "mAP50-95_mask": round(r["mAP50-95_mask"], 4),
            "mAP50-95_box": round(r["mAP50-95_box"], 4),
            "n_sem_escala": resumo["n_sem_escala"], "n_sem_fungo": resumo["n_sem_fungo"],
            "n_val": resumo["n_imagens"], "minutos_treino": round(r["segundos_treino"] / 60, 1),
        })
        print(f"    MAPE = {resumo['MAPE']:.2f}%   mAP50-95 (máscara) = {r['mAP50-95_mask']:.4f}   "
              f"treino: {r['segundos_treino'] / 60:.1f} min")
        if args.apagar_pesos:
            shutil.rmtree(r["pesos"].parent, ignore_errors=True)

    m = pd.read_csv(csv_metricas)
    print(f"\n{nome_cfg} ({len(m)} folds): MAPE = {m['MAPE'].mean():.2f} ± {m['MAPE'].std():.2f}%   "
          f"mAP50-95 (máscara) = {m['mAP50-95_mask'].mean():.4f} ± {m['mAP50-95_mask'].std():.4f}")


if __name__ == "__main__":
    main()
