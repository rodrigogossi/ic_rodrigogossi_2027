# Metodologia de preparação dos dados

Este documento descreve como o gabarito de área foi obtido, como o conjunto de
teste (20%) foi separado e como o treino foi aumentado. Todos os passos são
determinísticos e podem ser reproduzidos com os scripts em `scripts/`.

```
data/raw (542 imagens + labels YOLO do Label Studio)
   │
   │ 1. scripts/gerar_gabarito.py  → gabarito.csv (área em mm² por imagem)
   │
   │ 2. scripts/split_teste.py     → estratificação por área
   ├──────────► data/test          109 imagens (20,1%) — congelado
   ▼
data/train_base                    433 imagens (79,9%)
   │
   │ 3. scripts/augment.py         → 4 cópias por imagem, labels transformados
   ▼
data/train_aug                     2165 imagens
```

---

## 1. Gabarito de área (`gabarito.csv`)

### Origem

O gabarito é calculado a partir dos polígonos anotados no Label Studio, como
em 2026. O script `scripts/gerar_gabarito.py` é uma adaptação do
`gerar_gabarito.py` do repositório de 2026 (`ic_rodrigogossi_yollov8`), com o
mesmo cálculo, e reproduz o `gabarito.csv` de 2026 (diferença de 0,00 mm² nas
542 imagens).

### Cálculo

Cada imagem tem dois tipos de polígono:

| Classe | Objeto | Área real |
|---|---|---|
| 0 — `escala` | quadrado de papel de 4 cm × 4 cm | 1600 mm² (conhecida) |
| 1 — `fungo` | colônia fúngica | desconhecida |

Para cada imagem:

1. As coordenadas normalizadas (0–1) de cada polígono são multiplicadas pela
   largura e altura da imagem e truncadas para pixel inteiro.
2. A área de cada polígono em pixels é calculada com `cv2.contourArea`
   (fórmula do laço / *shoelace*).
3. **Escala:** se houver mais de um polígono, usa-se o maior.
   **Fungo:** soma-se a área de todos os polígonos de fungo.
4. Conversão por regra de três, usando a escala como régua dentro da imagem:

$$
\text{Área}_{\text{fungo}}\,(\text{mm}^2) = \text{Área}_{\text{fungo}}\,(\text{px}) \times \frac{1600\ \text{mm}^2}{\text{Área}_{\text{escala}}\,(\text{px})}
$$

5. O resultado é arredondado em 2 casas decimais.

### Distribuição

| | Área (mm²) |
|---|---|
| mínimo | 30,98 |
| p25 | 70,35 |
| mediana | 611,30 |
| p75 | 1994,38 |
| máximo | 5611,19 |

A distribuição é muito assimétrica. Metade das colônias tem menos de 611 mm²,
e cerca de 30% tem menos de 80 mm² (colônias logo após a inoculação). Por isso
um sorteio simples poderia deixar o teste com poucas colônias grandes ou poucas
pequenas, e a estratificação por área é necessária.

---

## 2. Separação do conjunto de teste (20%)

### Procedimento (`scripts/split_teste.py`)

1. **Associação da área:** cada imagem recebe sua área do `gabarito.csv`.
2. **Faixas de área:** as 542 imagens são divididas em 10 faixas com o mesmo
   número de imagens, os decis da área (`pandas.qcut`).
3. **Sorteio estratificado:** dentro de cada faixa, 20% das imagens são
   sorteadas para o teste (`sklearn.model_selection.train_test_split` com
   `stratify=faixa_area` e `random_state=42`).
4. **Isolamento:** as imagens de teste são copiadas para `data/test/` e o
   restante para `data/train_base/`. O manifesto `splits/split.csv` registra,
   para cada imagem, a área, a faixa, o conjunto e o hash SHA-256 da imagem e
   do label.

A estratificação usa **apenas a área da colônia**. Dia, código de tratamento
(`t2`…`t6`) e volume não entram no critério. A separação é feita por imagem,
então fotos da mesma placa em dias diferentes podem estar em conjuntos
diferentes.

### Resultado por faixa

| Faixa | Área (mm²) | Imagens | No teste | % |
|---|---|---|---|---|
| 1 | 30,98 – 56,86 | 55 | 11 | 20,0 |
| 2 | 56,89 – 66,20 | 54 | 11 | 20,4 |
| 3 | 66,53 – 78,20 | 54 | 11 | 20,4 |
| 4 | 78,92 – 253,18 | 54 | 11 | 20,4 |
| 5 | 264,67 – 608,33 | 54 | 11 | 20,4 |
| 6 | 614,27 – 1175,19 | 54 | 11 | 20,4 |
| 7 | 1176,27 – 1672,75 | 54 | 11 | 20,4 |
| 8 | 1676,33 – 2641,39 | 54 | 11 | 20,4 |
| 9 | 2641,74 – 3727,28 | 54 | 10 | 18,5 |
| 10 | 3755,45 – 5611,19 | 55 | 11 | 20,0 |
| **Total** | | **542** | **109** | **20,1** |

### Verificação da representatividade

Gerada automaticamente em `splits/representatividade.txt` e
`splits/representatividade.png`.

**Quantis da área (mm²):**

| | Completo | Treino | Teste |
|---|---|---|---|
| mínimo | 31,0 | 37,1 | 31,0 |
| p10 | 56,9 | 56,9 | 57,5 |
| p25 | 70,4 | 70,2 | 71,5 |
| mediana | 611,3 | 639,5 | 559,8 |
| p75 | 1994,4 | 1989,5 | 2118,1 |
| p90 | 3752,6 | 3749,8 | 3522,1 |
| máximo | 5611,2 | 5611,2 | 5365,3 |

**Testes estatísticos (treino × teste):**

| Variável | Teste | Resultado | Entra na estratificação? |
|---|---|---|---|
| Área | Kolmogorov-Smirnov | D = 0,040; **p = 0,998** | sim |
| Faixa de área | χ² | p = 1,000 | sim |
| Tratamento (`t2`…`t6`) | χ² | p = 0,515 | não |
| Volume (0–100 µL) | χ² | p = 0,442 | não |
| Dia (`d1`…`d5`) | χ² | p = 0,575 | não |

Um p > 0,05 indica que não há diferença detectável entre as distribuições de
treino e teste. Para a área, o p = 0,998 mostra que o teste reproduz a
distribuição de tamanhos do dataset completo. As demais variáveis não foram
controladas, mas também não diferem significativamente.

### Garantia de isolamento

- O teste é separado **antes** do aumento de dados, então nenhuma versão
  aumentada de uma imagem de teste existe no treino.
- `split_teste.py` recusa refazer a separação se `splits/split.csv` já
  existir, a menos que seja chamado com `--force`.
- `split_teste.py --verify` confere o SHA-256 de cada imagem e label de teste,
  aponta arquivos extras e verifica que nenhuma imagem de teste está no treino.

### Texto sugerido para o artigo

> O conjunto de teste (20%; n = 109) foi separado antes de qualquer aumento de
> dados, por amostragem estratificada segundo os decis da área da colônia
> (semente aleatória 42). A equivalência entre as distribuições de área dos
> conjuntos de treino e teste foi confirmada pelo teste de Kolmogorov-Smirnov
> (D = 0,040; p = 0,998). O conjunto de teste permaneceu isolado e não foi
> utilizado em nenhuma etapa de treinamento, validação ou ajuste de
> hiperparâmetros.

---

## 3. Aumento de dados (`scripts/augment.py`)

Aplicado **somente** a `data/train_base`. Cada imagem gera 4 cópias
aumentadas com a biblioteca Albumentations. Os labels são transformados
automaticamente: cada polígono é rasterizado como máscara, recebe a mesma
transformação da imagem e é reconvertido em polígono YOLO.

Como a área é medida pela razão `fungo / escala`, só são usadas
transformações geométricas que preservam essa razão:

| Tipo | Transformações |
|---|---|
| Geométricas | flip H/V, rotação 90°, rotação livre ±180°, escala uniforme 0,85–1,15, translação ±5% |
| Fotométricas | brilho/contraste, gamma, CLAHE, HSV leve, ruído gaussiano/ISO, blur, sombra, compressão JPEG |
| Não usadas | shear, perspectiva, distorção elástica, grid distortion |

Amostras em que alguma instância é cortada pela borda da imagem são sorteadas
de novo. As imagens aumentadas têm lado maior de 2048 px.

**Checagem:** a razão fungo/escala de cada label aumentado é comparada com a
do label original (`splits/augment.csv`). O erro foi de 0,16% na mediana, 0,82%
no p95 e 1,45% no máximo.

**K-Fold:** use `GroupKFold` com a coluna `origem` de `splits/augment.csv`,
para que as cópias aumentadas fiquem no mesmo fold que a imagem original.
