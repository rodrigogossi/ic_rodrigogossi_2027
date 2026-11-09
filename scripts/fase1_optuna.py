"""
Fase 1 — busca de hiperparâmetros com Optuna.

Cada tentativa treina 3 folds (30 épocas) e devolve a média do MAPE da área nas
imagens originais de validação. O TPE sugere a próxima combinação e o
MedianPruner interrompe tentativas que ficam acima da mediana após um fold.

O estudo fica em <saida>/optuna.db (SQLite). Se a execução cair, rode o mesmo
comando de novo: a busca continua de onde parou e a tentativa interrompida é
refeita com os mesmos parâmetros.

Uso:
    python scripts/fase1_optuna.py --imgsz 640                  # busca principal (30 tentativas)
    python scripts/fase1_optuna.py --imgsz 640 --segunda-fase   # 6 parâmetros extras, com os 6 principais fixos

Saídas em <saida>/fase1_<imgsz>/:
    tentativas.csv                      uma linha por tentativa (parâmetros, MAPE, mAP, falhas, tempo)
    busca_<imgsz>_tentativa_XXX/        treinos e áreas por imagem de cada fold
    melhores_parametros_principal.json  vencedora da busca principal
    melhores_parametros.json            configuração lida pela fase 2 (principal + segunda fase, se rodada)
"""
from __future__ import annotations

import argparse
import shutil
import time

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage, RetryFailedTrialCallback

import treino_comum as tc

N_INICIAIS_ALEATORIAS = 10


def sugerir_principal(trial: optuna.Trial) -> dict:
    opt = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    faixa_lr = {"SGD": (1e-3, 5e-2), "AdamW": (1e-4, 3e-3)}[opt]
    return {
        "modelo": trial.suggest_categorical("modelo", ["yolov8n-seg", "yolov8s-seg"]),
        "optimizer": opt,
        # nome separado por otimizador: o TPE trata como parâmetros condicionais
        "lr0": trial.suggest_float(f"lr0_{opt}", *faixa_lr, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "momentum": trial.suggest_float("momentum", 0.7, 0.98),
        "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
    }


def sugerir_segunda_fase(trial: optuna.Trial, base: dict) -> dict:
    return {
        **base,
        "lrf": trial.suggest_float("lrf", 0.01, 1.0),
        "warmup_epochs": trial.suggest_float("warmup_epochs", 0.0, 5.0),
        "hsv_s": trial.suggest_float("hsv_s", 0.0, 0.9),
        "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.9),
        "degrees": trial.suggest_float("degrees", 0.0, 45.0),
        "scale": trial.suggest_float("scale", 0.0, 0.9),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tc.args_comuns(ap)
    ap.add_argument("--tentativas", type=int, default=30, help="total de tentativas concluídas")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--segunda-fase", action="store_true",
                    help="busca lrf, warmup, hsv_s, hsv_v, degrees e scale com os 6 principais fixos")
    ap.add_argument("--apagar-pesos", action="store_true",
                    help="apaga os pesos de cada fold depois de medir a área (economiza disco)")
    args = ap.parse_args()

    pasta = args.saida / f"fase1_{args.imgsz}"
    pasta.mkdir(parents=True, exist_ok=True)
    nome_estudo = f"busca_{args.imgsz}" + ("_fase2" if args.segunda_fase else "")
    base = None
    if args.segunda_fase:
        _, base = tc.carregar_config(str(pasta / "melhores_parametros_principal.json"),
                                     args.imgsz, args.saida)

    df = tc.carregar_manifesto_treino()
    folds = list(tc.gerar_folds(df, args.folds))
    csv_tentativas = pasta / f"tentativas{'_fase2' if args.segunda_fase else ''}.csv"

    def objetivo(trial: optuna.Trial) -> float:
        params = sugerir_segunda_fase(trial, base) if base else sugerir_principal(trial)
        pasta_t = pasta / f"{nome_estudo}_tentativa_{trial.number:03d}"
        inicio = time.time()
        mapes, maps, areas = [], [], []
        for i, (nome, treino, val) in enumerate(folds):
            data_yaml = tc.escrever_dataset(pasta_t / nome / "dados", treino, val, args.max_val)
            r = tc.treinar(params, data_yaml, args.epochs, args.imgsz, pasta_t, nome, args)
            a = tc.medir_areas(r["pesos"], val, args.imgsz, args)
            a.insert(0, "fold", nome)
            a.to_csv(pasta_t / nome / "areas.csv", index=False)
            areas.append(a)
            mapes.append(a["erro_%"].mean())
            maps.append(r["mAP50-95_mask"])
            if args.apagar_pesos:
                shutil.rmtree(r["pesos"].parent, ignore_errors=True)

            media = sum(mapes) / len(mapes)
            trial.report(media, step=i)
            if trial.should_prune():
                trial.set_user_attr("folds_concluidos", i + 1)
                raise optuna.TrialPruned()

        resumo = tc.resumo_areas(pd.concat(areas))
        trial.set_user_attr("mAP50-95_mask", sum(maps) / len(maps))
        trial.set_user_attr("MAPE_por_fold", [round(m, 3) for m in mapes])
        trial.set_user_attr("horas", round((time.time() - inicio) / 3600, 2))
        for k, v in resumo.items():
            if k != "MAPE":
                trial.set_user_attr(k, v)
        return sum(mapes) / len(mapes)

    storage = RDBStorage(
        url=f"sqlite:///{args.saida / 'optuna.db'}",
        heartbeat_interval=60, grace_period=600,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=1),
    )
    estudo = optuna.create_study(
        study_name=nome_estudo, storage=storage, direction="minimize", load_if_exists=True,
        sampler=TPESampler(seed=tc.SEED, n_startup_trials=N_INICIAIS_ALEATORIAS),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    feitas = [t for t in estudo.trials
              if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    restantes = args.tentativas - len(feitas)
    print(f"Estudo {nome_estudo}: {len(feitas)} tentativas concluídas, {max(restantes, 0)} restantes.")
    if restantes > 0:
        estudo.optimize(objetivo, n_trials=restantes, gc_after_trial=True)

    estudo.trials_dataframe().to_csv(csv_tentativas, index=False)
    melhor = estudo.best_trial
    if base:
        params = {**base, **melhor.params}
    else:
        p = dict(melhor.params)
        params = {"modelo": p.pop("modelo"), "optimizer": p.pop("optimizer"),
                  "lr0": p.pop(f"lr0_{melhor.params['optimizer']}"), **p}
        tc.salvar_json(pasta / "melhores_parametros_principal.json", params)
    # melhores_parametros.json é o que a fase 2 lê: o da busca principal ou,
    # depois da segunda fase, a combinação das duas.
    tc.salvar_json(pasta / "melhores_parametros.json", params)
    print(f"\nMelhor tentativa: #{melhor.number}  MAPE = {melhor.value:.3f}%")
    print(f"Parâmetros salvos em {pasta / 'melhores_parametros.json'}:")
    for k, v in params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
