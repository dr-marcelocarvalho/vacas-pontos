# 🐄 Cows Challenge — Identificação Individual de Vacas via Pose Estimation

**Aluno:** Marcelo de Rezende Carvalho

Pipeline em Python para identificar vacas individualmente a partir de imagens de visão superior durante a ordenha. A ideia é combinar um modelo YOLO Pose (que detecta 8 keypoints anatômicos por animal) com classificadores clássicos de Machine Learning para resolver um problema que parece simples, mas tem bastante nuance: vacas da mesma raça, no mesmo ambiente, fotografadas do mesmo ângulo são difíceis de diferenciar.

---

## Sobre o Projeto

O dataset tem ~1500 imagens de 30 vacas capturadas por câmeras fixas instaladas no teto do estábulo durante a ordenha. O pipeline funciona assim:

1. Detecta a posição corporal da vaca com YOLO Pose (8 keypoints anatômicos)
2. Extrai 39 features geométricas — ângulos entre ossos, distâncias e proporções invariantes à escala
3. Treina Random Forest, SVM e Regressão Logística com StratifiedGroupKFold para evitar data leakage entre sessões de ordenha
4. Prediz qual vaca aparece numa imagem nova via CLI, com score de confiança e top-3

---

## Pré-requisitos

- Python 3.10+
- GPU NVIDIA com CUDA (recomendado — testado numa RTX 3070 Ti 8GB)
- 16 GB de RAM no mínimo (64 GB para processamento sem solavanco)
- Windows 10/11, Linux ou macOS

---

## Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd cow-challenge
```

### 2. Crie e ative um ambiente virtual

```bash
# Windows
python -m venv venv
venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install ultralytics numpy opencv-python matplotlib pillow pyyaml pandas scikit-learn joblib seaborn scipy tabulate
```

O `ultralytics` já puxa o PyTorch com suporte a CUDA automaticamente, mas vale verificar depois:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Estrutura do Projeto

```
cow-challenge/
├── data/
│   ├── raw_images/                   # ~1500 imagens JPEG brutas
│   ├── dataset_classificação/        # 30 pastas (uma por vaca), ~50 imagens cada
│   ├── subset_yolo_pose/             # gerado no Passo A.2
│   └── processed/                    # gerado no Passo C (features.csv)
├── Key_points/                       # 1030 JSONs de anotações do Label Studio
├── src/
│   ├── core_utils.py
│   ├── validate.py                   # Passo A.1
│   ├── make_subset.py                # Passo A.2
│   ├── train_pose.py                 # Passo B.1
│   ├── evaluate_pose.py              # Passo B.2
│   ├── extract_features.py           # Passo C
│   ├── analyze_features.py           # Passo D
│   ├── train_classifier.py           # Passo E
│   ├── evaluate_classifier.py        # Passo F.1
│   └── predict.py                    # Passo F.2
├── outputs/
│   ├── models/
│   ├── reports/
│   └── figures/
└── requirements.txt
```

---

## Rodando o Pipeline

Execute cada passo em ordem. Todos os scripts assumem que você está na raiz do projeto com o ambiente virtual ativo.

### A — Preparação do Dataset

#### A.1 — Validar anotações

```bash
python src/validate.py
```

Lê os 1030 JSONs de `Key_points/` e verifica se cada anotação tem bounding box e os 8 keypoints esperados. Resultado em `outputs/reports/validation_report.json`. Roda em menos de 10 segundos.

#### A.2 — Montar o subset para o YOLO

```bash
python src/make_subset.py
```

Filtra anotações válidas, embaralha (seed=42), seleciona 150 imagens e divide 80/20. Cria a estrutura de pastas que o Ultralytics espera, com `data.yaml` incluído. Leva uns 30 segundos.

---

### B — Treino e Avaliação do YOLO Pose

#### B.1 — Treinar

```bash
python src/train_pose.py
```

Parte do `yolo11n-pose.pt` (baixado automaticamente) e treina por 100 épocas com `imgsz=640`. Detecta GPU sozinho. Salva o melhor modelo em `outputs/models/best_pose.pt`.

Tempo estimado: 15–25 min com GPU, 2–4h no CPU.

#### B.2 — Avaliar

```bash
python src/evaluate_pose.py
```

Roda `model.val()` na partição de validação e grava mAP50, mAP50-95, Precision e Recall em `outputs/reports/metrics.json`. Menos de 1 minuto.

---

### C — Extração de Features

```bash
python src/extract_features.py
```

Infere keypoints em todas as ~1500 imagens e calcula por imagem:
- 16 coordenadas brutas (X, Y de cada keypoint)
- 5 ângulos (cosseno entre trios de pontos anatômicos)
- 9 distâncias euclidianas
- 9 proporções normalizadas pelo comprimento do dorso

Saída: `data/processed/features.csv`. Leva de 5 a 15 minutos dependendo da GPU.

---

### D — Análise Descritiva

```bash
python src/analyze_features.py
```

Gera histogramas, heatmap de correlação, boxplots por estação e câmera, e um pairplot das principais features. Tudo vai para `outputs/figures/`. Uns 2 minutos.

---

### E — Treino dos Classificadores

```bash
python src/train_classifier.py
```

Esse é o passo mais pesado. O script:

1. Percorre as 30 pastas de `data/dataset_classificação/`
2. Infere keypoints com YOLO em cada imagem
3. Aplica data augmentation (9 cópias por imagem com ruído gaussiano σ=2 nas coordenadas)
4. Extrai features invariantes à escala (ângulos + proporções)
5. Cria `session_id` por hash de câmera + estação + data para agrupar sessões
6. Treina RF (200 árvores), SVM (RBF) e Regressão Logística com StratifiedGroupKFold k=5
7. Calcula Acurácia, F1-macro, Top-1/3/5

Saída: `outputs/models/cow_classifier.joblib` e matrizes de confusão em `outputs/figures/`. Tempo estimado: 30–60 min.

---

### F — Avaliação Final e Predição

#### F.1 — Relatório de avaliação

```bash
python src/evaluate_classifier.py
```

Gera gráfico de acurácia por vaca e curvas Precision-Recall (top 15). Precisa re-inferir o YOLO no dataset de classificação, então leva uns 10–20 min.

#### F.2 — Identificar uma vaca

```bash
python src/predict.py --image caminho/para/imagem.jpg
```

Exemplo:

```bash
python src/predict.py --image "data/raw_images/20260101_041009_baia19_IPC1.jpg"
```

Saída no terminal:

```
═══════════════════════════════════════════════════════
  🐄 RESULTADO DA IDENTIFICAÇÃO
═══════════════════════════════════════════════════════
  📸 Imagem: 20260101_041009_baia19_IPC1.jpg

  🥇 Cow 1325        ██████████████████░░░░░░░░░░░░  15.2%
  🥈 Cow 1288        █████████████░░░░░░░░░░░░░░░░░  10.8%
  🥉 Cow 1362        ████████████░░░░░░░░░░░░░░░░░░   9.5%

  ⚠️  Confiança moderada. Verificar manualmente.
═══════════════════════════════════════════════════════
```

Menos de 5 segundos.

---

## Keypoints Anatômicos

| Índice | Nome | Descrição |
|--------|------|-----------|
| 0 | withers | Cernelha |
| 1 | back | Dorso |
| 2 | hook up | Tuberosidade coxal superior |
| 3 | hook down | Tuberosidade coxal inferior |
| 4 | hip | Garupa |
| 5 | tail head | Base da cauda |
| 6 | pin up | Tuberosidade isquiática sup. |
| 7 | pin down | Tuberosidade isquiática inf. |

Esqueleto:

```
withers(0) → back(1) → hip(4) → hook_up(2)
                         ↓    → hook_down(3)
                         ↓    → tail_head(5) → pin_up(6)
                                               → pin_down(7)
```

---

## Features Geométricas

| Tipo | Qtd | Descrição |
|------|-----|-----------|
| Coordenadas | 16 | X, Y de cada keypoint |
| Ângulos | 5+1 | Ângulos entre trios de pontos |
| Distâncias | 9 | Distâncias euclidianas no esqueleto |
| Proporções | 9 | Distâncias normalizadas pelo dorso |

Para classificação, só ângulos e proporções são usados — eles não dependem da distância da câmera nem da escala da imagem.

---

## Classificadores e Validação

| Modelo | Parâmetros |
|--------|------------|
| Random Forest | n_estimators=200 |
| SVM | kernel="rbf" |
| Logistic Regression | max_iter=1000 |

A validação usa **StratifiedGroupKFold k=5** com `session_id` como grupo. Isso garante que imagens da mesma sessão de ordenha não apareçam em treino e teste ao mesmo tempo — sem isso, o modelo aprenderia a reconhecer sessões, não vacas.

O data augmentation com ruído gaussiano σ=2 nas coordenadas ajuda a simular variações de detecção sem precisar coletar mais dados.

---

## Solução de Problemas

**"CUDA not available"**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Erro de memória na GPU**
Abra `src/train_pose.py` e mude `batch=-1` para `batch=4` ou `batch=8`.

**Imagens não encontradas pelo make_subset.py**
As imagens precisam estar em `data/raw_images/` e os JSONs em `Key_points/`.

**"ModuleNotFoundError"**
O ambiente virtual provavelmente não está ativo:
```bash
# Windows
venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate
```

---

## Hardware Testado

| Componente | Configuração |
|------------|-------------|
| GPU | NVIDIA GeForce RTX 3070 Ti 8GB |
| CPU | AMD Ryzen 9 5900X 12-Core |
| RAM | 64 GB DDR4 |
| OS | Windows 10/11 |
| Python | 3.10+ |
| Tempo total | ~1h30 para o pipeline completo |