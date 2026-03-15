"""
predict.py — Predição Single-Image (CLI).

Recebe uma imagem de vaca, detecta keypoints com YOLO Pose,
extrai features geométricas, e prediz cow_id com confiança.

Uso:
    python src/predict.py --image caminho/para/imagem.jpg
"""

import argparse
import sys
import warnings
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.core_utils import TARGET_KPS, OUTPUTS_DIR
from src.extract_features import extract_features_from_keypoints

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
#  DISPLAY ANSI
# ═══════════════════════════════════════════════════════════════════

def colored(text: str, color: str) -> str:
    """Aplica cor ANSI a um texto."""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def render_bar(percentage: float, width: int = 30) -> str:
    """Renderiza uma barra de progresso ANSI."""
    filled = int(width * percentage / 100.0)
    empty = width - filled

    if percentage >= 50:
        bar_color = "\033[92m"  # verde
    elif percentage >= 25:
        bar_color = "\033[93m"  # amarelo
    else:
        bar_color = "\033[91m"  # vermelho

    bar = f"{bar_color}{'█' * filled}\033[90m{'░' * empty}\033[0m"
    return bar


def display_predictions(predictions: list, image_name: str) -> None:
    """Exibe as top-3 predições com barras ANSI."""
    print()
    print(colored("═" * 55, "bold"))
    print(colored("  🐄 RESULTADO DA IDENTIFICAÇÃO", "bold"))
    print(colored("═" * 55, "bold"))
    print(f"  📸 Imagem: {colored(image_name, 'cyan')}")
    print()

    for rank, (cow_id, prob) in enumerate(predictions[:3], 1):
        pct = prob * 100
        bar = render_bar(pct)

        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"

        label = f"  {medal} Cow {cow_id}"
        print(f"{label:<22} {bar}  {colored(f'{pct:.1f}%', 'bold')}")

    print()

    # Confiança geral
    top_prob = predictions[0][1] * 100
    if top_prob >= 70:
        msg = colored("  ✅ Alta confiança na identificação!", "green")
    elif top_prob >= 40:
        msg = colored("  ⚠️  Confiança moderada. Verificar manualmente.", "yellow")
    else:
        msg = colored("  ❌ Baixa confiança. Imagem pode não ser adequada.", "red")

    print(msg)
    print(colored("═" * 55, "bold"))
    print()


# ═══════════════════════════════════════════════════════════════════
#  PIPELINE DE PREDIÇÃO
# ═══════════════════════════════════════════════════════════════════

def predict_single_image(
    image_path: str,
    yolo_model_path: str = None,
    classifier_path: str = None,
) -> list:
    """
    Prediz cow_id a partir de uma única imagem.

    Retorna lista de (cow_id, probabilidade) ordenada decrescente.
    """
    # Paths padrão
    if yolo_model_path is None:
        yolo_model_path = str(OUTPUTS_DIR / "models" / "best_pose.pt")
    if classifier_path is None:
        classifier_path = str(OUTPUTS_DIR / "models" / "cow_classifier.joblib")

    # Verificar existência dos modelos
    if not Path(yolo_model_path).exists():
        raise FileNotFoundError(f"Modelo YOLO não encontrado: {yolo_model_path}")
    if not Path(classifier_path).exists():
        raise FileNotFoundError(f"Classificador não encontrado: {classifier_path}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # 1. YOLO Pose inference
    yolo = YOLO(yolo_model_path)
    results = yolo(image_path, verbose=False)

    if not results or len(results) == 0:
        raise ValueError("Nenhuma detecção encontrada na imagem.")

    result = results[0]

    if result.keypoints is None or len(result.keypoints) == 0:
        raise ValueError("Nenhum keypoint detectado na imagem.")
    if result.boxes is None or len(result.boxes) == 0:
        raise ValueError("Nenhuma bounding box detectada na imagem.")

    # Melhor detecção
    confs = result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    print(f"  📦 BBox confiança: {best_conf:.4f}")

    kps_data = result.keypoints.data.cpu().numpy()
    if best_idx >= kps_data.shape[0]:
        raise ValueError("Índice de keypoint inválido.")

    kps_single = kps_data[best_idx]
    kps_xy = kps_single[:, :2]
    kps_conf = kps_single[:, 2]

    print(f"  🦴 Keypoints detectados: {kps_xy.shape[0]}/8")
    for i, name in enumerate(TARGET_KPS):
        status = "✅" if kps_conf[i] > 0.3 else "❌"
        print(f"     {status} {name}: conf={kps_conf[i]:.3f}")

    # 2. Feature extraction
    features = extract_features_from_keypoints(kps_xy)
    if features is None:
        raise ValueError("Não foi possível extrair features dos keypoints.")

    # 3. Carregar classificador
    package = joblib.load(classifier_path)
    clf = package["model"]
    scaler = package["scaler"]
    le = package["label_encoder"]
    feature_columns = package["feature_columns"]

    # 4. Preparar vetor de features
    feat_vector = np.array([[features.get(col, 0.0) for col in feature_columns]])
    feat_scaled = scaler.transform(feat_vector)

    # 5. Predição
    probas = clf.predict_proba(feat_scaled)[0]

    # Ordenar
    ranked = sorted(
        zip(le.classes_, probas),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked


# ═══════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="🐄 Cows Challenge — Identificação de Vaca por Imagem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python src/predict.py --image data/raw_images/20260101_041009_baia19_IPC1.jpg
  python src/predict.py --image foto_vaca.jpg
        """,
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Caminho para a imagem da vaca",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=None,
        help="Caminho para o modelo YOLO Pose (padrão: outputs/models/best_pose.pt)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Caminho para o classificador (padrão: outputs/models/cow_classifier.joblib)",
    )

    args = parser.parse_args()

    try:
        predictions = predict_single_image(
            args.image,
            yolo_model_path=args.yolo_model,
            classifier_path=args.classifier,
        )
        display_predictions(predictions, Path(args.image).name)

    except FileNotFoundError as e:
        print(colored(f"\n  ❌ {e}", "red"))
        sys.exit(1)
    except ValueError as e:
        print(colored(f"\n  ⚠️  {e}", "yellow"))
        sys.exit(1)
    except Exception as e:
        print(colored(f"\n  💥 Erro inesperado: {e}", "red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
