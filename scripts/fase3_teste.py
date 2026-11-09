"""
Fase 3 — modelo final e avaliação ÚNICA no conjunto de teste isolado.

1. Confere a integridade do teste (sha256, como em split_teste.py --verify).
2. Treina com todo o data/train_aug (2165 imagens), 100 épocas, com a
   configuração vencedora da fase 2.
3. Mede a área nas 109 imagens de data/test e calcula MAPE, R², MAE, RMSE,
   Pearson, Bland-Altman, IC 95% por bootstrap e MAPE por faixa de área.
4. Calcula o mAP da segmentação no teste.

O teste só pode ser usado uma vez: o script grava AVALIADO.lock e se recusa a
rodar de novo sem --force. Nenhum resultado do teste deve voltar para escolhas
de modelo ou de hiperparâmetros.

Uso:
    python scripts/fase3_teste.py --imgsz 640                    # usa fase2_640/vencedora.json
    python scripts/fase3_teste.py --imgsz 640 --config arq.json

Saídas em <saida>/fase3_<imgsz>/:
    resultado_teste.txt / .json   métricas finais (o número que vai para o artigo)
    areas_teste.csv               área prevista × gabarito por imagem
    validacao_area_teste.png      dispersão e Bland-Altman
    final/weights/last.pt         modelo final
"""
from __future__ import annotations

import argparse
import datetime
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import treino_comum as tc

N_BOOTSTRAP = 2000


def bootstrap_ic(real: np.ndarray, prev: np.ndarray, erro: np.ndarray) -> dict:
    rng = np.random.default_rng(tc.SEED)
    idx = rng.integers(0, len(real), size=(N_BOOTSTRAP, len(real)))
    mape = erro[idx].mean(axis=1)
    ok = ~np.isnan(prev)
    r2 = []
    for i in idx:
        i = i[ok[i]]
        r, p = real[i], prev[i]
        r2.append(1 - ((r - p) ** 2).sum() / ((r - r.mean()) ** 2).sum())
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"MAPE_IC95": q(mape), "R2_IC95": q(np.array(r2))}


def figura(df, caminho) -> None:
    d = df[df["falha"] != "sem_escala"]
    real, prev = d["real_mm2"].to_numpy(), d["prevista_mm2"].to_numpy()
    media, dif = (real + prev) / 2, prev - real
    vies, dp = dif.mean(), dif.std(ddof=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    lim = [0, max(real.max(), prev.max()) * 1.05]
    ax[0].scatter(real, prev, s=14, alpha=0.7)
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlim=lim, ylim=lim, xlabel="Área do gabarito (mm²)", ylabel="Área prevista (mm²)",
              title="Teste: previsto × gabarito")
    ax[1].scatter(media, dif, s=14, alpha=0.7)
    for y, ls in ((vies, "-"), (vies + 1.96 * dp, "--"), (vies - 1.96 * dp, "--")):
        ax[1].axhline(y, color="k", ls=ls, lw=1)
    ax[1].set(xlabel="Média (mm²)", ylabel="Previsto − gabarito (mm²)", title="Bland-Altman")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.args_comuns(ap)
    ap.add_argument("--config", default=None,
                    help="padrao, otimizado ou .json (padrão: fase2_<imgsz>/vencedora.json)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="avalia de novo mesmo com AVALIADO.lock")
    args = ap.parse_args()

    pasta = args.saida / f"fase3_{args.imgsz}"
    trava = pasta / "AVALIADO.lock"
    if trava.exists() and not args.force:
        sys.exit(f"O teste já foi avaliado ({trava.read_text().strip()}). "
                 "Ele deve ser usado uma única vez; use --force só se tiver certeza.")

    sys.path.insert(0, str(tc.ROOT / "scripts"))
    import split_teste
    split_teste.verify()

    config = args.config or str(args.saida / f"fase2_{args.imgsz}" / "vencedora.json")
    nome_cfg, params = tc.carregar_config(config, args.imgsz, args.saida)
    tc.salvar_json(pasta / "config.json", {"origem": config, "params": params, "fixos": tc.FIXOS,
                                           "epochs": args.epochs, "imgsz": args.imgsz})

    # Treino com todo o treino aumentado. A Ultralytics exige um conjunto de
    # validação; usamos as originais do treino (só para a validação final
    # interna, sem efeito na escolha). O teste não entra aqui.
    df = tc.carregar_manifesto_treino()
    treino = df["imagem"].tolist()
    originais = df.loc[~df["aumentada"], "imagem"].tolist()
    data_yaml = tc.escrever_dataset(pasta / "dados", treino, originais, args.max_val)
    print(f"Treinando o modelo final ({nome_cfg}, {len(treino)} imagens, {args.epochs} épocas)...")
    r = tc.treinar(params, data_yaml, args.epochs, args.imgsz, pasta, "final", args, plots=True)

    # Avaliação única no teste.
    pasta.mkdir(parents=True, exist_ok=True)
    trava.write_text(f"avaliado em {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    testes = sorted((tc.TEST / "images").glob("*.jpg"))
    areas = tc.medir_areas(r["pesos"], testes, args.imgsz, args)
    areas.to_csv(pasta / "areas_teste.csv", index=False)

    from ultralytics import YOLO
    yaml_teste = tc.escrever_dataset(pasta / "dados_teste", testes, testes, args.max_val)
    val = YOLO(str(r["pesos"])).val(data=str(yaml_teste), split="val", imgsz=args.imgsz,
                                    batch=tc.FIXOS["batch"], device=args.device, plots=True,
                                    project=str(pasta), name="val_teste", exist_ok=True, verbose=False)

    ok = areas[areas["falha"] != "sem_escala"]
    real, prev = ok["real_mm2"].to_numpy(), ok["prevista_mm2"].to_numpy()
    dif = prev - real
    resumo = tc.resumo_areas(areas)
    res = {
        "imgsz": args.imgsz, "config": nome_cfg, "n_teste": len(areas),
        "MAPE": round(resumo["MAPE"], 4),
        **bootstrap_ic(areas["real_mm2"].to_numpy(), areas["prevista_mm2"].to_numpy(),
                       areas["erro_%"].to_numpy()),
        "R2": round(float(1 - (dif ** 2).sum() / ((real - real.mean()) ** 2).sum()), 4),
        "MAE_mm2": round(float(np.abs(dif).mean()), 3),
        "RMSE_mm2": round(float(np.sqrt((dif ** 2).mean())), 3),
        "Pearson_r": round(float(stats.pearsonr(real, prev)[0]), 4),
        "BlandAltman_vies_mm2": round(float(dif.mean()), 3),
        "BlandAltman_LoA_mm2": [round(float(dif.mean() - 1.96 * dif.std(ddof=1)), 3),
                                round(float(dif.mean() + 1.96 * dif.std(ddof=1)), 3)],
        "mAP50_mask": round(float(val.seg.map50), 4),
        "mAP50-95_mask": round(float(val.seg.map), 4),
        **{k: v for k, v in resumo.items() if k not in ("MAPE", "n_imagens")},
    }
    tc.salvar_json(pasta / "resultado_teste.json", res)
    texto = "\n".join(f"{k:<24} {v}" for k, v in res.items())
    (pasta / "resultado_teste.txt").write_text(texto + "\n")
    figura(areas, pasta / "validacao_area_teste.png")
    print("\nResultado no teste isolado:\n" + texto)


if __name__ == "__main__":
    main()
