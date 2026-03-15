"""
make_subset.py — Criação do subset YOLO Pose.

Filtra anotações válidas, seleciona 150 imagens (seed=42),
divide 80/20 em train/val e escreve a hierarquia Ultralytics
em data/subset_yolo_pose/ com data.yaml.
"""

import json
import random
import shutil
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core_utils import (
    TARGET_KPS,
    KEYPOINTS_DIR,
    DATA_DIR,
    RAW_IMAGES_DIR,
    load_annotation_json,
    validate_annotation,
    parse_annotation_results,
    resolve_image_path,
)


SUBSET_SIZE = 150
TRAIN_RATIO = 0.8
SEED = 42


def build_yolo_pose_label(
    bbox: dict,
    keypoints: dict,
    class_id: int = 0,
) -> str:
    """
    Gera uma linha de label YOLO Pose:
    class x_center y_center width height px1 py1 v1 px2 py2 v2 ...

    Todos os valores normalizados [0,1].
    """
    parts = [str(class_id)]

    # BBox centro
    parts.append(f"{bbox['x']:.6f}")
    parts.append(f"{bbox['y']:.6f}")
    parts.append(f"{bbox['width']:.6f}")
    parts.append(f"{bbox['height']:.6f}")

    # Keypoints na ordem TARGET_KPS
    for kp_name in TARGET_KPS:
        if kp_name in keypoints:
            kp = keypoints[kp_name]
            parts.append(f"{kp['x']:.6f}")
            parts.append(f"{kp['y']:.6f}")
            parts.append(f"{int(kp['visibility'])}")
        else:
            parts.append("0.000000")
            parts.append("0.000000")
            parts.append("0")

    return " ".join(parts)


def main() -> None:
    print("=" * 60)
    print("  🐄 CRIAÇÃO DO SUBSET YOLO POSE — Cows Challenge")
    print("=" * 60)

    # 1. Coletar anotações válidas
    json_files = sorted(KEYPOINTS_DIR.glob("*"))
    json_files = [f for f in json_files if f.is_file() and f.name != ".DS_Store"]

    valid_entries: list = []

    print(f"  Analisando {len(json_files)} anotações...")

    for jf in json_files:
        try:
            image_ref, results = load_annotation_json(jf)
            if not results:
                continue

            validation = validate_annotation(results)
            if not validation["valid"]:
                continue

            # Resolver caminho da imagem
            if not image_ref:
                continue

            image_path = resolve_image_path(image_ref, RAW_IMAGES_DIR)
            if image_path is None or not image_path.exists():
                continue

            # Parsear bbox e keypoints
            bbox, keypoints = parse_annotation_results(results)
            if bbox is None:
                continue

            valid_entries.append({
                "json_file": jf,
                "image_path": image_path,
                "bbox": bbox,
                "keypoints": keypoints,
            })

        except Exception as e:
            print(f"  ⚠️  Erro em {jf.name}: {e}")
            continue

    print(f"  ✅ Anotações válidas com imagem: {len(valid_entries)}")

    # 2. Shuffle e selecionar subset
    random.seed(SEED)
    random.shuffle(valid_entries)

    subset_size = min(SUBSET_SIZE, len(valid_entries))
    subset = valid_entries[:subset_size]
    print(f"  📦 Subset selecionado: {subset_size} imagens")

    # 3. Split train/val
    train_count = int(subset_size * TRAIN_RATIO)
    train_set = subset[:train_count]
    val_set = subset[train_count:]

    print(f"  🏋️  Train: {len(train_set)} | Val: {len(val_set)}")

    # 4. Criar hierarquia de diretórios
    subset_dir = DATA_DIR / "subset_yolo_pose"

    # Limpar diretório anterior se existir
    if subset_dir.exists():
        shutil.rmtree(subset_dir)

    for split in ["train", "val"]:
        (subset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (subset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 5. Copiar imagens e criar labels
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        for entry in split_data:
            img_src = entry["image_path"]
            img_dst = subset_dir / "images" / split_name / img_src.name

            # Copiar imagem
            shutil.copy2(img_src, img_dst)

            # Gerar label
            label_line = build_yolo_pose_label(entry["bbox"], entry["keypoints"])
            label_path = subset_dir / "labels" / split_name / (img_src.stem + ".txt")
            label_path.write_text(label_line, encoding="utf-8")

    # 6. Criar data.yaml
    data_yaml = {
        "path": str(subset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "cow"},
        "nc": 1,
        "kpt_shape": [8, 3],
    }

    yaml_path = subset_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print()
    print("=" * 60)
    print("  ✅ SUBSET CRIADO COM SUCESSO")
    print(f"  📁 {subset_dir}")
    print(f"  📄 {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
