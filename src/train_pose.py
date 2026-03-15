"""
train_pose.py — Treino do Modelo YOLO Pose.

Carrega yolo11n-pose.pt como base, treina por 100 épocas
no subset_yolo_pose e salva o melhor modelo em outputs/models/best_pose.pt.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.core_utils import DATA_DIR, OUTPUTS_DIR


def get_device() -> str:
    """Detecta o melhor dispositivo disponível."""
    import torch

    if torch.cuda.is_available():
        dev = "cuda"
        print(f"  🖥️  GPU detectada: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
        print("  🍎 Apple MPS detectado")
    else:
        dev = "cpu"
        print("  💻 Usando CPU")
    return dev


def main() -> None:
    print("=" * 60)
    print("  🐄 TREINAMENTO YOLO POSE — Cows Challenge")
    print("=" * 60)

    data_yaml = DATA_DIR / "subset_yolo_pose" / "data.yaml"
    if not data_yaml.exists():
        print("  ❌ data.yaml não encontrado. Execute make_subset.py primeiro.")
        sys.exit(1)

    device = get_device()

    # Carregar modelo base
    print("  📥 Carregando modelo base yolo11n-pose.pt ...")
    model = YOLO("yolo11n-pose.pt")

    # Treinar
    print(f"  🏋️  Iniciando treino — 100 épocas, imgsz=640")
    print(f"  📁 Dataset: {data_yaml}")
    print()

    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=-1,           # Auto-batch
        device=device,
        project=str(OUTPUTS_DIR / "runs"),
        name="pose_train",
        exist_ok=True,
        verbose=True,
        seed=42,
    )

    # Copiar melhor modelo
    models_dir = OUTPUTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = models_dir / "best_pose.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print()
        print("=" * 60)
        print(f"  ✅ Melhor modelo salvo em: {best_dst}")
        print("=" * 60)
    else:
        print("  ⚠️  best.pt não encontrado nos resultados do treino.")
        # Tenta o last.pt como fallback
        last_src = Path(results.save_dir) / "weights" / "last.pt"
        if last_src.exists():
            shutil.copy2(last_src, best_dst)
            print(f"  📦 Usando last.pt como fallback: {best_dst}")


if __name__ == "__main__":
    main()
