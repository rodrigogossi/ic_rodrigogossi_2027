# IC 2027 — Evolução da medição automática de colônias fúngicas

> Rodrigo Gossi · Iniciação Científica
> Continuação de [ic_rodrigogossi_yollov8](../ic_rodrigogossi_yollov8) (YOLOv8n-seg, 2026).

## Pipeline de dados

> Metodologia completa (gabarito, estratificação e aumento):
> [docs/metodologia_dados.md](docs/metodologia_dados.md)
>
> Hiperparâmetros do YOLO e plano de busca com Optuna (métrica otimizada:
> MAPE da área):
> [docs/hiperparametros_optuna.md](docs/hiperparametros_optuna.md) ([PDF](docs/hiperparametros_optuna.pdf))

```
data/raw  (542 imagens + labels YOLO do Label Studio)
   │
   │  scripts/gerar_gabarito.py — gabarito.csv: área (mm²) a partir dos polígonos
   │  scripts/split_teste.py   — estratificado pela área da colônia
   ├──────────────► data/test        109 imagens (20,1%)  (CONGELADO, nunca usar no treino)
   ▼
data/train_base   433 imagens
   │
   │  scripts/augment.py       — Albumentations; labels transformados automaticamente
   ▼
data/train_aug    433 originais + 1732 aumentadas = 2165 imagens
```

### Representatividade do teste

O teste é separado **antes** do aumento, estratificado apenas pela **área da
colônia**: as imagens são divididas em 10 faixas de área (decis da área do
`gabarito.csv`, calculada dos polígonos do Label Studio por `scripts/gerar_gabarito.py`)
e 20% de cada faixa vai para o teste (seed 42). Assim o
teste cobre de colônias recém-inoculadas (~31 mm²) até as maiores (~5400 mm²).

Resultado (`splits/representatividade.txt` / `.png`):

| | Completo | Treino | Teste |
|---|---|---|---|
| mín (mm²) | 31,0 | 37,1 | 31,0 |
| p25 | 70,4 | 70,2 | 71,5 |
| mediana | 611,3 | 639,5 | 559,8 |
| p75 | 1994,4 | 1989,5 | 2118,1 |
| p90 | 3752,6 | 3749,8 | 3522,1 |
| máx | 5611,2 | 5611,2 | 5365,3 |

- Cada decil de área tem 18,5–20,4% das imagens no teste.
- Kolmogorov-Smirnov treino × teste: D = 0,040, **p = 0,998**, ou seja, sem
  diferença detectável na distribuição de área.
- Só para informação, como não entram na estratificação: tratamento
  (χ² p = 0,52), volume (p = 0,44) e dia (p = 0,58) também não diferem
  significativamente entre treino e teste.

No K-Fold do treino use `GroupKFold` com a coluna `origem` de
`splits/augment.csv`, para que as cópias aumentadas fiquem sempre no mesmo
fold que a imagem original.

### Aumento de dados

O cálculo de área usa a razão `área_fungo / área_escala`, então só são usadas
transformações geométricas que **preservam essa razão**:

| Tipo | Transformações |
|---|---|
| Geométricas | flip H/V, rotação 90°, rotação livre ±180°, escala uniforme 0,85–1,15, translação ±5% |
| Fotométricas | brilho/contraste, gamma, CLAHE, HSV leve, ruído gaussiano/ISO, blur, sombra, compressão JPEG |
| **Não usadas** | shear, perspectiva, elástica, grid distortion (distorcem a escala 4×4 cm) |

Os polígonos são rasterizados como máscaras, transformados junto com a imagem
e reconvertidos em polígonos YOLO. Amostras em que alguma instância é cortada
pela borda são sorteadas de novo. As imagens aumentadas têm lado maior de 2048 px.

Checagem (em `splits/augment.csv`): erro na razão fungo/escala após a
transformação — mediana 0,16%, p95 0,82%, máx 1,45%.

## Uso

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# copiar o dataset bruto para data/raw/{images,labels}
.venv/bin/python scripts/gerar_gabarito.py         # gera gabarito.csv (área em mm²)
.venv/bin/python scripts/split_teste.py            # cria data/test, data/train_base e o relatório
.venv/bin/python scripts/split_teste.py --verify   # confere integridade do teste (sha256)
.venv/bin/python scripts/augment.py --n 4          # cria data/train_aug
.venv/bin/python scripts/visualizar_labels.py      # mosaicos de conferência em checks/
```

As imagens não são versionadas (`data/` está no `.gitignore`). A separação é
reproduzível pela seed fixa e documentada em `splits/split.csv`, que registra
o hash SHA-256 de cada arquivo de teste.

## Treinamento

O desenho das três fases está em
[docs/hiperparametros_optuna.md](docs/hiperparametros_optuna.md). Cada fase é
um script em `scripts/`; o código comum (folds, treino, medição de área) fica
em `scripts/treino_comum.py`.

### Preparar a máquina de treino

```bash
git clone https://github.com/rodrigogossi/ic_rodrigogossi_2027.git && cd ic_rodrigogossi_2027
python3 -m venv .venv && .venv/bin/pip install -r requirements-treino.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Copiar para a máquina as pastas `data/train_aug/` e `data/test/` (não estão no
git) e conferir o teste antes de começar:

```bash
.venv/bin/python scripts/split_teste.py --verify
```

Os treinos levam horas ou dias: rode dentro de `tmux` ou `screen` para não
perder a execução se a conexão cair. Todos os scripts retomam de onde pararam
se forem chamados de novo com os mesmos argumentos.

### 0. Medir o tempo real

Antes da busca, treinar 1 fold com poucas épocas para refazer o orçamento:

```bash
.venv/bin/python scripts/fase2_avaliacao.py --imgsz 640 --config padrao --max-folds 1 --epochs 3 --saida runs/tempo
```

O tempo por época sai em `runs/tempo/fase2_640/padrao/metricas_folds.csv`
(`minutos_treino` / 3).

### 1. Busca com Optuna

```bash
.venv/bin/python scripts/fase1_optuna.py --imgsz 640                  # 30 tentativas × 3 folds × 30 épocas
.venv/bin/python scripts/fase1_optuna.py --imgsz 640 --segunda-fase   # opcional, se sobrar tempo
```

O Optuna minimiza o MAPE da área nas imagens originais de validação de cada
fold. O estudo fica em `runs/optuna.db`; a configuração vencedora vai para
`runs/fase1_640/melhores_parametros.json`.

### 2. Otimizado × padrão

```bash
.venv/bin/python scripts/fase2_avaliacao.py --imgsz 640 --config padrao
.venv/bin/python scripts/fase2_avaliacao.py --imgsz 640 --config otimizado
.venv/bin/python scripts/fase2_comparar.py --imgsz 640
```

Repeated K-Fold 3×5 com 100 épocas, os mesmos folds nas duas configurações.
A comparação (Wilcoxon e t pareados, MAPE por faixa de área) fica em
`runs/fase2_640/comparacao.txt`, e a configuração com menor MAPE em
`runs/fase2_640/vencedora.json`.

### 3. Modelo final e teste

```bash
.venv/bin/python scripts/fase3_teste.py --imgsz 640
```

Treina com todo o `data/train_aug` e avalia **uma única vez** nas 109 imagens
de teste (MAPE com IC 95%, R², MAE, RMSE, Bland-Altman, MAPE por faixa, mAP).
O script grava `runs/fase3_640/AVALIADO.lock` e não roda de novo sem `--force`.

Para 1024 px, os mesmos comandos com `--imgsz 1024`.

### Detalhes de implementação

- **Folds:** K-Fold sobre as imagens originais, estratificado pela faixa de
  área; as cópias aumentadas acompanham a origem (equivale ao `GroupKFold`
  por `origem`). A validação usa só as originais.
- **Pesos:** a validação durante o treino fica desligada (`val=False`) e
  todas as fases usam os pesos da última época (`last.pt`). A escolha é feita
  pelo MAPE, e a fase 3 não tem conjunto de validação para escolher época.
- **Área:** mesmos parâmetros de inferência de 2026 (conf 0,20, iou 0,60,
  `retina_masks`). Imagem sem escala ou sem fungo detectados conta como erro
  de 100%.
- **Resultados** ficam em `runs/` (fora do git). Para rodar um teste do
  código sem a GPU, use `--saida` para outra pasta.

## Gabarito (`gabarito.csv`)

| Coluna | Descrição |
|---|---|
| `identificador` | Nome da imagem sem extensão |
| `Area(mm²)` | Área da colônia calculada dos polígonos do Label Studio: `área_fungo_px × 1600 / área_escala_px` |

## Classes

| ID | Nome | Descrição |
|---|---|---|
| 0 | `escala` | Marcador físico 4 cm × 4 cm (1600 mm²) |
| 1 | `fungo` | Colônia fúngica |
