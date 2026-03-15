"""
evaluate_pose.py — Avaliação do Modelo YOLO Pose.

Roda model.val() na partição de validação e salva as métricas
(mAP50, mAP50-95 para BBox e Keypoints) em outputs/reports/metrics.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.core_utils import DATA_DIR, OUTPUTS_DIR


def main() -> None:
    print("=" * 60)
    print("  🐄 AVALIAÇÃO YOLO POSE — Cows Challenge")
    print("=" * 60)

    model_path = OUTPUTS_DIR / "models" / "best_pose.pt"
    data_yaml = DATA_DIR / "subset_yolo_pose" / "data.yaml"

    if not model_path.exists():
        print(f"  ❌ Modelo não encontrado: {model_path}")
        print("  Execute train_pose.py primeiro.")
        sys.exit(1)

    if not data_yaml.exists():
        print(f"  ❌ data.yaml não encontrado: {data_yaml}")
        sys.exit(1)

    print(f"  📥 Modelo: {model_path}")
    print(f"  📁 Dataset: {data_yaml}")
    print()

    # Carregar e avaliar
    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_yaml),
        imgsz=640,
        verbose=True,
    )

    # Extrair métricas
    metrics = {
        "bbox": {
            "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else None,
            "mAP50_95": float(results.box.map) if hasattr(results.box, 'map') else None,
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else None,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else None,
        },
        "keypoints": {
            "mAP50": float(results.pose.map50) if hasattr(results, 'pose') and hasattr(results.pose, 'map50') else None,
            "mAP50_95": float(results.pose.map) if hasattr(results, 'pose') and hasattr(results.pose, 'map') else None,
            "precision": float(results.pose.mp) if hasattr(results, 'pose') and hasattr(results.pose, 'mp') else None,
            "recall": float(results.pose.mr) if hasattr(results, 'pose') and hasattr(results.pose, 'mr') else None,
        },
    }

    # Exibir
    print()
    print("=" * 60)
    print("  📊 MÉTRICAS DE AVALIAÇÃO")
    print("=" * 60)
    print()
    print("  Bounding Box:")
    print(f"    mAP@50:    {metrics['bbox']['mAP50']:.4f}" if metrics['bbox']['mAP50'] else "    mAP@50:    N/A")
    print(f"    mAP@50-95: {metrics['bbox']['mAP50_95']:.4f}" if metrics['bbox']['mAP50_95'] else "    mAP@50-95: N/A")
    print(f"    Precision: {metrics['bbox']['precision']:.4f}" if metrics['bbox']['precision'] else "    Precision: N/A")
    print(f"    Recall:    {metrics['bbox']['recall']:.4f}" if metrics['bbox']['recall'] else "    Recall:    N/A")
    print()
    print("  Keypoints:")
    print(f"    mAP@50:    {metrics['keypoints']['mAP50']:.4f}" if metrics['keypoints']['mAP50'] else "    mAP@50:    N/A")
    print(f"    mAP@50-95: {metrics['keypoints']['mAP50_95']:.4f}" if metrics['keypoints']['mAP50_95'] else "    mAP@50-95: N/A")
    print(f"    Precision: {metrics['keypoints']['precision']:.4f}" if metrics['keypoints']['precision'] else "    Precision: N/A")
    print(f"    Recall:    {metrics['keypoints']['recall']:.4f}" if metrics['keypoints']['recall'] else "    Recall:    N/A")
    print()

    # Salvar relatório
    reports_dir = OUTPUTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "metrics.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  📁 Métricas salvas em: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
