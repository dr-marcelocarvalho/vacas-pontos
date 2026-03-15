"""
evaluate_classifier.py — Relatório Final do Classificador.

Carrega o pacote cow_classifier.joblib, gera gráficos de
distribuição de acurácia por vaca e curvas Precision-Recall.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.core_utils import DATA_DIR, OUTPUTS_DIR, parse_filename_metadata
from src.extract_features import extract_features_from_keypoints
from src.train_classifier import make_session_id

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BBOX_CONF_THRESHOLD = 0.5
KP_CONF_THRESHOLD = 0.3


def main() -> None:
    print("=" * 60)
    print("  🐄 AVALIAÇÃO FINAL DO CLASSIFICADOR — Cows Challenge")
    print("=" * 60)

    # Carregar pacote
    package_path = OUTPUTS_DIR / "models" / "cow_classifier.joblib"
    if not package_path.exists():
        print(f"  ❌ Pacote não encontrado: {package_path}")
        print("  Execute train_classifier.py primeiro.")
        sys.exit(1)

    package = joblib.load(package_path)
    clf = package["model"]
    scaler = package["scaler"]
    le = package["label_encoder"]
    feature_columns = package["feature_columns"]
    results_summary = package.get("results_summary", {})

    figures_dir = OUTPUTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Carregar YOLO
    model_path = OUTPUTS_DIR / "models" / "best_pose.pt"
    if not model_path.exists():
        print(f"  ❌ Modelo YOLO não encontrado: {model_path}")
        sys.exit(1)

    yolo_model = YOLO(str(model_path))

    # ─── 1. Coletar dados de teste ───
    classif_dir = DATA_DIR / "dataset_classificação"
    cow_folders = sorted([
        d for d in classif_dir.iterdir()
        if d.is_dir() and d.name != ".DS_Store"
    ])

    print(f"  🐄 Coletando dados de {len(cow_folders)} vacas...")

    all_features = []
    all_labels = []

    for cow_folder in cow_folders:
        cow_id = cow_folder.name
        image_files = sorted([
            f for f in cow_folder.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        for img_path in image_files:
            results = yolo_model(str(img_path), verbose=False)
            if not results or len(results) == 0:
                continue

            result = results[0]
            if result.keypoints is None or result.boxes is None:
                continue
            if len(result.keypoints) == 0 or len(result.boxes) == 0:
                continue

            confs = result.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            best_conf = float(confs[best_idx])

            if best_conf < BBOX_CONF_THRESHOLD:
                continue

            kps_data = result.keypoints.data.cpu().numpy()
            if best_idx >= kps_data.shape[0]:
                continue

            kps_single = kps_data[best_idx]
            kps_xy = kps_single[:, :2]
            kps_conf = kps_single[:, 2]

            if kps_xy.shape[0] != 8 or np.any(kps_conf < KP_CONF_THRESHOLD):
                continue

            features = extract_features_from_keypoints(kps_xy)
            if features is None:
                continue

            feat_vector = [features.get(col, 0.0) for col in feature_columns]
            all_features.append(feat_vector)
            all_labels.append(cow_id)

    if not all_features:
        print("  ❌ Nenhuma amostra coletada para avaliação.")
        sys.exit(1)

    X = np.array(all_features)
    y_raw = np.array(all_labels)
    y = le.transform(y_raw)

    X_scaled = scaler.transform(X)

    y_pred = clf.predict(X_scaled)
    y_proba = clf.predict_proba(X_scaled)

    print(f"  📊 Amostras: {len(X)}")
    print()

    # ─── 2. Acurácia por Vaca ───
    print("  📊 Gerando distribuição de acurácia por vaca...")

    per_cow_acc = {}
    for cow_label in le.classes_:
        cow_encoded = le.transform([cow_label])[0]
        mask = y == cow_encoded
        if mask.sum() == 0:
            continue
        cow_acc = accuracy_score(y[mask], y_pred[mask])
        per_cow_acc[cow_label] = cow_acc

    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(16, 6))
    cows_sorted = sorted(per_cow_acc.keys(), key=lambda x: per_cow_acc[x], reverse=True)
    accs_sorted = [per_cow_acc[c] for c in cows_sorted]

    colors = ["#2ecc71" if a >= 0.7 else "#e74c3c" if a < 0.4 else "#f39c12" for a in accs_sorted]

    bars = ax.bar(range(len(cows_sorted)), accs_sorted, color=colors, edgecolor="white")
    ax.set_xticks(range(len(cows_sorted)))
    ax.set_xticklabels(cows_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Acurácia")
    ax.set_title("Acurácia por Vaca Individual", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(accs_sorted), color="navy", linestyle="--", alpha=0.7,
               label=f"Média: {np.mean(accs_sorted):.2f}")
    ax.legend()

    # Adicionar valores nas barras
    for bar, acc in zip(bars, accs_sorted):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.0%}",
            ha="center", va="bottom", fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(figures_dir / "accuracy_per_cow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── 3. Curvas Precision-Recall ───
    print("  📈 Gerando curvas Precision-Recall...")

    n_classes = len(le.classes_)
    y_bin = label_binarize(y, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotar curva para cada classe (com seleção das top 10 e pior 5)
    ap_scores = {}
    for i, cow_label in enumerate(le.classes_):
        if y_bin.shape[1] > i:
            ap = average_precision_score(y_bin[:, i], y_proba[:, i])
            ap_scores[cow_label] = ap

    # Ordenar por AP
    sorted_cows = sorted(ap_scores.items(), key=lambda x: x[1], reverse=True)

    # Plotar top 10
    cmap = plt.cm.get_cmap("tab20", min(15, n_classes))
    for plot_idx, (cow_label, ap) in enumerate(sorted_cows[:15]):
        i = list(le.classes_).index(cow_label)
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(recall, precision, lw=1.5, alpha=0.7,
                color=cmap(plot_idx), label=f"Cow {cow_label} (AP={ap:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall por Vaca (Top 15)", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    fig.savefig(figures_dir / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ─── 4. Relatório Final ───
    print()
    print("=" * 60)
    print("  📋 RELATÓRIO FINAL")
    print("=" * 60)

    overall_acc = accuracy_score(y, y_pred)
    overall_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"  Acurácia Global: {overall_acc:.4f}")
    print(f"  F1 Macro:        {overall_f1:.4f}")
    print(f"  Média por Vaca:  {np.mean(accs_sorted):.4f}")
    print()

    if results_summary:
        print("  Cross-Validation (da etapa de treino):")
        for clf_name, metrics in results_summary.items():
            print(f"    {clf_name}: Acc={metrics['accuracy']:.4f} | "
                  f"Top-3={metrics['top3']:.4f} | Top-5={metrics['top5']:.4f}")

    # Salvar relatório markdown
    report_lines = [
        "# Relatório Final — Classificação de Vacas\n",
        f"**Acurácia Global:** {overall_acc:.4f}\n",
        f"**F1 Macro:** {overall_f1:.4f}\n",
        f"**Média por Vaca:** {np.mean(accs_sorted):.4f}\n",
        f"**Total de Amostras:** {len(X)}\n",
        f"**Total de Classes:** {n_classes}\n",
        "",
        "## Figuras",
        "- `accuracy_per_cow.png`",
        "- `precision_recall_curves.png`",
        "- `confusion_matrix_*.png`",
    ]

    report_path = figures_dir / "evaluation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print()
    print(f"  📁 Figuras e relatório em: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
