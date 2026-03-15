"""
train_classifier.py — Treinamento dos Classificadores de Identificação de Vaca.

Percorre data/dataset_classificação/, infere keypoints com YOLO Pose,
aplica data augmentation via ruído gaussiano, e treina Random Forest,
SVM e Logistic Regression com StratifiedGroupKFold (n=5).
Salva modelo + scaler + encoder em outputs/models/cow_classifier.joblib.
"""

import sys
import hashlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.core_utils import (
    TARGET_KPS,
    DATA_DIR,
    OUTPUTS_DIR,
    parse_filename_metadata,
)
from src.extract_features import extract_features_from_keypoints

warnings.filterwarnings("ignore")

# Configurações
N_AUGMENTATIONS = 9
NOISE_STD = 2.0
BBOX_CONF_THRESHOLD = 0.5
KP_CONF_THRESHOLD = 0.3
RANDOM_SEED = 42
N_FOLDS = 5

sns.set_theme(style="whitegrid")


# ═══════════════════════════════════════════════════════════════════
#  UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════

def make_session_id(meta: dict) -> str:
    """
    Cria um hash de sessão baseado em cam + station + date
    para agrupar imagens da mesma sessão.
    """
    cam = meta.get("cam") or "unknown"
    station = meta.get("station") or "unknown"
    date = meta.get("date") or "unknown"
    raw = f"{cam}_{station}_{date}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def get_scale_invariant_columns(feature_dict: dict) -> List[str]:
    """Retorna apenas colunas de ângulos e proporções (invariantes à escala)."""
    return [
        k for k in feature_dict.keys()
        if k.startswith("angle_") or k.startswith("ratio_")
    ]


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """Calcula top-k accuracy a partir de probabilidades."""
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    return correct / len(y_true)


# ═══════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  🐄 TREINAMENTO DE CLASSIFICADORES — Cows Challenge")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # ─── 1. Descobrir vacas ───
    classif_dir = DATA_DIR / "dataset_classificação"
    if not classif_dir.exists():
        print(f"  ❌ Diretório não encontrado: {classif_dir}")
        sys.exit(1)

    cow_folders = sorted([
        d for d in classif_dir.iterdir()
        if d.is_dir() and d.name != ".DS_Store"
    ])

    cow_ids = [f.name for f in cow_folders]
    print(f"  🐄 Vacas encontradas: {len(cow_ids)}")
    print(f"  IDs: {', '.join(cow_ids)}")

    # ─── 2. Carregar modelo YOLO ───
    model_path = OUTPUTS_DIR / "models" / "best_pose.pt"
    if not model_path.exists():
        print(f"  ❌ Modelo YOLO não encontrado: {model_path}")
        print("  Execute train_pose.py primeiro.")
        sys.exit(1)

    print(f"  📥 Modelo YOLO: {model_path}")
    model = YOLO(str(model_path))

    # ─── 3. Extrair features com augmentation ───
    print()
    print("  🔬 Extraindo features com data augmentation...")
    print(f"  Augmentações por imagem: {N_AUGMENTATIONS}")
    print(f"  Ruído Gaussiano σ: {NOISE_STD}")
    print()

    rows: List[Dict] = []
    cow_labels: List[str] = []
    session_ids: List[str] = []
    total_images = 0
    skipped_images = 0

    for cow_folder in cow_folders:
        cow_id = cow_folder.name
        image_files = sorted([
            f for f in cow_folder.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        for img_path in image_files:
            total_images += 1

            # Inferência YOLO
            results = model(str(img_path), verbose=False)
            if not results or len(results) == 0:
                skipped_images += 1
                continue

            result = results[0]
            if result.keypoints is None or result.boxes is None:
                skipped_images += 1
                continue
            if len(result.keypoints) == 0 or len(result.boxes) == 0:
                skipped_images += 1
                continue

            # Melhor detecção
            confs = result.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            best_conf = float(confs[best_idx])

            if best_conf < BBOX_CONF_THRESHOLD:
                skipped_images += 1
                continue

            kps_data = result.keypoints.data.cpu().numpy()
            if best_idx >= kps_data.shape[0]:
                skipped_images += 1
                continue

            kps_single = kps_data[best_idx]  # (8, 3)
            kps_xy = kps_single[:, :2].copy()
            kps_conf = kps_single[:, 2]

            # Verificar keypoints
            if kps_xy.shape[0] != 8 or np.any(kps_conf < KP_CONF_THRESHOLD):
                skipped_images += 1
                continue

            # Metadados e session_id
            meta = parse_filename_metadata(img_path.name)
            sid = make_session_id(meta)

            # Original + N augmentations
            for aug_idx in range(1 + N_AUGMENTATIONS):
                if aug_idx == 0:
                    kps_augmented = kps_xy.copy()
                else:
                    noise = np.random.normal(0, NOISE_STD, kps_xy.shape)
                    kps_augmented = kps_xy + noise

                features = extract_features_from_keypoints(kps_augmented)
                if features is None:
                    continue

                features["cow_id"] = cow_id
                features["filename"] = img_path.name
                features["augmentation"] = aug_idx
                features["session_id"] = sid
                features["bbox_conf"] = best_conf

                rows.append(features)
                cow_labels.append(cow_id)
                session_ids.append(sid)

        if (cow_folders.index(cow_folder) + 1) % 5 == 0:
            print(f"    Processadas {cow_folders.index(cow_folder) + 1}/{len(cow_folders)} vacas...")

    print()
    print(f"  📊 Total de imagens: {total_images}")
    print(f"  ⏭️  Descartadas: {skipped_images}")
    print(f"  📈 Amostras geradas (orig + aug): {len(rows)}")

    if len(rows) == 0:
        print("  ❌ Nenhuma amostra gerada. Verifique o modelo e as imagens.")
        sys.exit(1)

    # ─── 4. Montar DataFrame ───
    df = pd.DataFrame(rows)

    # Selecionar apenas features invariantes à escala
    sample_features = extract_features_from_keypoints(np.zeros((8, 2)))
    if sample_features is None:
        print("  ❌ Erro ao determinar colunas de features.")
        sys.exit(1)

    scale_inv_cols = get_scale_invariant_columns(sample_features)
    # Verificar quais existem no DataFrame
    scale_inv_cols = [c for c in scale_inv_cols if c in df.columns]

    print(f"  🎯 Features selecionadas (invariantes): {len(scale_inv_cols)}")

    X = df[scale_inv_cols].values
    y_raw = df["cow_id"].values
    groups = df["session_id"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"  🏷️  Classes: {len(le.classes_)}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ─── 5. StratifiedGroupKFold ───
    print()
    print("=" * 60)
    print(f"  🔄 CROSS-VALIDATION — StratifiedGroupKFold (k={N_FOLDS})")
    print("=" * 60)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_SEED
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }

    results_summary: Dict[str, Dict] = {}

    for clf_name, clf_template in classifiers.items():
        print(f"\n  ── {clf_name} ──")

        fold_accs = []
        fold_f1s = []
        fold_top3 = []
        fold_top5 = []
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X_scaled, y, groups)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone do classificador
            import sklearn.base
            clf = sklearn.base.clone(clf_template)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            fold_accs.append(acc)
            fold_f1s.append(f1)

            # Top-k
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_test)
                fold_top3.append(top_k_accuracy(y_test, y_proba, k=3))
                fold_top5.append(top_k_accuracy(y_test, y_proba, k=5))

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

            print(f"    Fold {fold_idx + 1}: Acc={acc:.4f} | F1={f1:.4f}")

        mean_acc = np.mean(fold_accs)
        mean_f1 = np.mean(fold_f1s)
        mean_top3 = np.mean(fold_top3) if fold_top3 else 0
        mean_top5 = np.mean(fold_top5) if fold_top5 else 0

        print(f"\n    📊 Média — Acc: {mean_acc:.4f} | F1: {mean_f1:.4f}")
        print(f"    🎯 Top-1: {mean_acc:.4f} | Top-3: {mean_top3:.4f} | Top-5: {mean_top5:.4f}")

        results_summary[clf_name] = {
            "accuracy": mean_acc,
            "f1_macro": mean_f1,
            "top1": mean_acc,
            "top3": mean_top3,
            "top5": mean_top5,
            "fold_accs": fold_accs,
        }

        # Confusion Matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=ax,
        )
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f"Confusion Matrix — {clf_name}")
        plt.tight_layout()

        figures_dir = OUTPUTS_DIR / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / f"confusion_matrix_{clf_name}.png", dpi=150)
        plt.close(fig)

    # ─── 6. Treinar modelo final e salvar ───
    print()
    print("=" * 60)
    print("  💾 TREINANDO MODELO FINAL (Random Forest)")
    print("=" * 60)

    final_clf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
    )
    final_clf.fit(X_scaled, y)

    # Salvar pacote
    models_dir = OUTPUTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    package = {
        "model": final_clf,
        "scaler": scaler,
        "label_encoder": le,
        "feature_columns": scale_inv_cols,
        "results_summary": results_summary,
    }

    package_path = models_dir / "cow_classifier.joblib"
    joblib.dump(package, package_path)

    print(f"  ✅ Pacote salvo em: {package_path}")
    print()

    # Resumo final
    print("=" * 60)
    print("  📊 RESUMO DE RESULTADOS")
    print("=" * 60)
    for clf_name, metrics in results_summary.items():
        print(f"  {clf_name}:")
        print(f"    Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
        print(f"    Top-1: {metrics['top1']:.4f} | Top-3: {metrics['top3']:.4f} | Top-5: {metrics['top5']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
