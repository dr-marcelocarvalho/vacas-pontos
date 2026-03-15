"""
extract_features.py — Extração de Features Geométricas.

Usa o modelo best_pose.pt para inferir keypoints em todas as imagens
de raw_images/, calcula 39 features geométricas (16 coords brutas,
5 ângulos, 9 distâncias, 9 proporções) e salva em data/processed/features.csv.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.core_utils import (
    TARGET_KPS,
    SKELETON,
    RAW_IMAGES_DIR,
    DATA_DIR,
    OUTPUTS_DIR,
    parse_filename_metadata,
)


# ═══════════════════════════════════════════════════════════════════
#  FUNÇÕES MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════════

def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distância euclidiana entre dois pontos 2D."""
    return float(np.linalg.norm(a - b))


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Ângulo (em graus) no ponto B formado pelos vetores BA e BC.
    Usa cosseno inverso clamp.
    """
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


# ═══════════════════════════════════════════════════════════════════
#  EXTRAÇÃO DE FEATURES DE UM CONJUNTO DE KEYPOINTS
# ═══════════════════════════════════════════════════════════════════

# Índices para referência rápida
KP_IDX = {name: i for i, name in enumerate(TARGET_KPS)}

# Pares de distância do esqueleto (contíguos) + extras
DISTANCE_PAIRS = [
    ("withers", "back"),
    ("back", "hip"),
    ("hip", "hook up"),
    ("hip", "hook down"),
    ("hip", "tail head"),
    ("tail head", "pin up"),
    ("tail head", "pin down"),
    # Extras
    ("withers", "hip"),
    ("withers", "tail head"),
]

# Trios de ângulo (ponto central é o segundo)
ANGLE_TRIPLES = [
    ("withers", "back", "hip"),
    ("back", "hip", "tail head"),
    ("hip", "tail head", "pin up"),
    ("hip", "tail head", "pin down"),
    ("hook up", "hip", "hook down"),
]

# Nome clean para coluna
def _dist_col(a: str, b: str) -> str:
    return f"dist_{a.replace(' ', '_')}_{b.replace(' ', '_')}"

def _angle_col(a: str, b: str, c: str) -> str:
    return f"angle_{a.replace(' ', '_')}_{b.replace(' ', '_')}_{c.replace(' ', '_')}"

def _ratio_col(numerator: str, denominator: str = "spine") -> str:
    return f"ratio_{numerator.replace(' ', '_')}_per_{denominator.replace(' ', '_')}"


def extract_features_from_keypoints(
    kps: np.ndarray,
) -> Optional[Dict[str, float]]:
    """
    Dado um array (8, 2) de coordenadas de keypoints (na ordem TARGET_KPS),
    calcula as 39 features.

    Retorna None se dados insuficientes.
    """
    if kps.shape != (8, 2):
        return None

    features: Dict[str, float] = {}

    # --- 16 coordenadas brutas ---
    for i, name in enumerate(TARGET_KPS):
        col_base = name.replace(" ", "_")
        features[f"{col_base}_x"] = float(kps[i, 0])
        features[f"{col_base}_y"] = float(kps[i, 1])

    # --- 5 ângulos ---
    for a_name, b_name, c_name in ANGLE_TRIPLES:
        a_pt = kps[KP_IDX[a_name]]
        b_pt = kps[KP_IDX[b_name]]
        c_pt = kps[KP_IDX[c_name]]
        col = _angle_col(a_name, b_name, c_name)
        features[col] = angle_between(a_pt, b_pt, c_pt)

    # Ângulo extra: pin_up - tail_head - pin_down
    a_pt = kps[KP_IDX["pin up"]]
    b_pt = kps[KP_IDX["tail head"]]
    c_pt = kps[KP_IDX["pin down"]]
    # Nota: já está coberto por "hip-tail head-pin down", mas adicionamos pin_up-tail-pin_down
    # Verificando: os 5 ângulos do prompt são:
    # 1. withers-back-hip  ✓
    # 2. back-hip-tail     ✓
    # 3. hip-tail-pin_up   → já temos hip-tail_head-pin up  ✓
    # 4. hip-tail-pin_down → já temos hip-tail_head-pin down ✓ (mas é o 4o)
    # 5. hook_up-hip-hook_down ✓
    # E pin_up-tail-pin_down como extra
    features[_angle_col("pin up", "tail head", "pin down")] = angle_between(
        kps[KP_IDX["pin up"]], kps[KP_IDX["tail head"]], kps[KP_IDX["pin down"]]
    )

    # --- 9 distâncias ---
    distances: Dict[str, float] = {}
    for a_name, b_name in DISTANCE_PAIRS:
        d = distance(kps[KP_IDX[a_name]], kps[KP_IDX[b_name]])
        col = _dist_col(a_name, b_name)
        distances[col] = d
        features[col] = d

    # --- 9 proporções geométricas ---
    # Comprimento do dorso: withers → tail head
    spine_length = distances.get(_dist_col("withers", "tail head"), 1e-8)
    if spine_length < 1e-8:
        spine_length = 1e-8

    # 7 distâncias do esqueleto normalizadas pelo comprimento do dorso
    skeleton_dist_names = [
        _dist_col("withers", "back"),
        _dist_col("back", "hip"),
        _dist_col("hip", "hook up"),
        _dist_col("hip", "hook down"),
        _dist_col("hip", "tail head"),
        _dist_col("tail head", "pin up"),
        _dist_col("tail head", "pin down"),
        _dist_col("withers", "hip"),
    ]

    for d_name in skeleton_dist_names:
        ratio = distances.get(d_name, 0) / spine_length
        features[_ratio_col(d_name, "spine")] = ratio

    # Proporção lateral: hook_up-hook_down / pin_up-pin_down
    dist_hook = distance(kps[KP_IDX["hook up"]], kps[KP_IDX["hook down"]])
    dist_pin = distance(kps[KP_IDX["pin up"]], kps[KP_IDX["pin down"]])
    features["ratio_hook_width_per_pin_width"] = dist_hook / max(dist_pin, 1e-8)

    return features


# ═══════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  🐄 EXTRAÇÃO DE FEATURES — Cows Challenge")
    print("=" * 60)

    model_path = OUTPUTS_DIR / "models" / "best_pose.pt"
    if not model_path.exists():
        print(f"  ❌ Modelo não encontrado: {model_path}")
        print("  Execute train_pose.py primeiro.")
        sys.exit(1)

    print(f"  📥 Modelo: {model_path}")
    model = YOLO(str(model_path))

    # Coletar imagens
    image_extensions = {".jpg", ".jpeg", ".png"}
    images = sorted([
        f for f in RAW_IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    print(f"  📸 Total de imagens: {len(images)}")
    print()

    rows: List[Dict] = []
    skipped = 0

    for idx, img_path in enumerate(images):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  Processando {idx + 1}/{len(images)}...")

        # Inferência
        results = model(str(img_path), verbose=False)

        if not results or len(results) == 0:
            skipped += 1
            continue

        result = results[0]

        # Verificar se há detecções com keypoints
        if result.keypoints is None or len(result.keypoints) == 0:
            skipped += 1
            continue

        # Pegar a detecção com maior confiança de BBox
        if result.boxes is None or len(result.boxes) == 0:
            skipped += 1
            continue

        confs = result.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])

        if best_conf < 0.3:
            skipped += 1
            continue

        # Extrair keypoints da melhor detecção
        kps_data = result.keypoints.data.cpu().numpy()

        if best_idx >= kps_data.shape[0]:
            skipped += 1
            continue

        kps_single = kps_data[best_idx]  # Shape: (8, 3) — x, y, conf

        # Verificar confiança dos keypoints
        kps_xy = kps_single[:, :2]  # (8, 2)
        kps_conf = kps_single[:, 2]  # (8,)

        # Verificar se todos os 8 kps têm confiança > 0.3
        if np.any(kps_conf < 0.3):
            skipped += 1
            continue

        # Extrair features
        features = extract_features_from_keypoints(kps_xy)
        if features is None:
            skipped += 1
            continue

        # Metadados da imagem
        meta = parse_filename_metadata(img_path.name)
        features["filename"] = img_path.name
        features["bbox_conf"] = best_conf
        features.update({f"meta_{k}": v for k, v in meta.items()})

        rows.append(features)

    # Salvar CSV
    print()
    print(f"  ✅ Features extraídas: {len(rows)}")
    print(f"  ⏭️  Ignoradas: {skipped}")

    if rows:
        df = pd.DataFrame(rows)

        output_dir = DATA_DIR / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "features.csv"

        df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"  📁 CSV salvo em: {output_path}")
        print(f"  📊 Shape: {df.shape}")
    else:
        print("  ⚠️  Nenhuma feature extraída. Verifique o modelo e as imagens.")

    print("=" * 60)


if __name__ == "__main__":
    main()
