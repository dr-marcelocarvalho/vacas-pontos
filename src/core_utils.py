"""
core_utils.py — Núcleo do projeto Cows Challenge.

Contém constantes, parsers de nome de arquivo, parsers de anotação
Label Studio e funções de validação de keypoints.
"""

import re
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import unquote


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════

TARGET_KPS: List[str] = [
    "withers",
    "back",
    "hook up",
    "hook down",
    "hip",
    "tail head",
    "pin up",
    "pin down",
]

# Conexões entre keypoints (índices em TARGET_KPS) para desenhar o esqueleto
SKELETON: List[Tuple[int, int]] = [
    (0, 1),   # withers  → back
    (1, 4),   # back     → hip
    (4, 2),   # hip      → hook up
    (4, 3),   # hip      → hook down
    (4, 5),   # hip      → tail head
    (5, 6),   # tail head→ pin up
    (5, 7),   # tail head→ pin down
]

# Diretórios-base (relativos à raiz do projeto)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
KEYPOINTS_DIR = PROJECT_ROOT / "Key_points"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ═══════════════════════════════════════════════════════════════════
#  PARSE DE NOME DE ARQUIVO
# ═══════════════════════════════════════════════════════════════════

# Padrão 1: cow_id_YYYY_MM_DD_HH_MM_SS_cam_ID_station_id.jpg
_PAT_COW_ID = re.compile(
    r"^(?P<cow_id>\d+)_"
    r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_"
    r"(?P<cam>[^_]+)_(?P<station>.+?)\.jpg$",
    re.IGNORECASE,
)

# Padrão 2: cam_ID_XX_YYYYMMDDHHMMSS_station_id_cam_ID.jpg
#   ex: RLC1_00_20260101060438_baia10_RLC1.jpg
_PAT_CAM_FIRST = re.compile(
    r"^(?P<cam>[A-Za-z0-9]+)_\d+_"
    r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
    r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})_"
    r"(?P<station>[^_]+)_(?P<cam2>[A-Za-z0-9]+)"
    r"(?:\(\d+\))?\.jpg$",
    re.IGNORECASE,
)

# Padrão 3: YYYYMMDD_HHMMSS_station_id_cam_ID.jpg
#   ex: 20260101_041009_baia19_IPC1.jpg
_PAT_DATE_FIRST = re.compile(
    r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_"
    r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})_"
    r"(?P<station>[^_]+)_(?P<cam>[A-Za-z0-9]+)\.jpg$",
    re.IGNORECASE,
)


def parse_filename_metadata(filename: str) -> Dict[str, Optional[str]]:
    """
    Extrai metadados de um nome de arquivo de imagem.

    Retorna dict com: cow_id, date (YYYY-MM-DD), time (HH:MM:SS), station, cam.
    Campos não encontrados ficam como None.
    """
    basename = Path(filename).name

    # Tenta Padrão 1 (com cow_id explícito)
    m = _PAT_COW_ID.match(basename)
    if m:
        return {
            "cow_id": m.group("cow_id"),
            "date": f"{m.group('year')}-{m.group('month')}-{m.group('day')}",
            "time": f"{m.group('hour')}:{m.group('minute')}:{m.group('second')}",
            "station": m.group("station"),
            "cam": m.group("cam"),
        }

    # Tenta Padrão 2 (cam_XX_...)
    m = _PAT_CAM_FIRST.match(basename)
    if m:
        return {
            "cow_id": None,
            "date": f"{m.group('year')}-{m.group('month')}-{m.group('day')}",
            "time": f"{m.group('hour')}:{m.group('minute')}:{m.group('second')}",
            "station": m.group("station"),
            "cam": m.group("cam"),
        }

    # Tenta Padrão 3 (YYYYMMDD_HHMMSS_...)
    m = _PAT_DATE_FIRST.match(basename)
    if m:
        return {
            "cow_id": None,
            "date": f"{m.group('year')}-{m.group('month')}-{m.group('day')}",
            "time": f"{m.group('hour')}:{m.group('minute')}:{m.group('second')}",
            "station": m.group("station"),
            "cam": m.group("cam"),
        }

    # Nenhum padrão casou
    return {
        "cow_id": None,
        "date": None,
        "time": None,
        "station": None,
        "cam": None,
    }


# ═══════════════════════════════════════════════════════════════════
#  RESOLUÇÃO DE CAMINHOS DE IMAGEM
# ═══════════════════════════════════════════════════════════════════

def resolve_image_path(image_ref_from_json: str, raw_dir: Path = RAW_IMAGES_DIR) -> Optional[Path]:
    """
    Dada a referência de imagem extraída do JSON Label Studio
    (campo task.data.img), resolve o caminho real no disco.

    O JSON costuma vir no formato:
        /data/local-files/?d=Users%5C...%5Cnome_imagem.jpg

    Extrai apenas o basename e busca em raw_dir.
    """
    # Decodifica URL-encoding
    decoded = unquote(image_ref_from_json)
    # Pega só o nome do arquivo (último segmento após qualquer separador)
    basename = Path(decoded.replace("\\", "/")).name

    candidate = raw_dir / basename
    if candidate.exists():
        return candidate

    # Busca case-insensitive
    for f in raw_dir.iterdir():
        if f.name.lower() == basename.lower():
            return f

    return None


# ═══════════════════════════════════════════════════════════════════
#  PARSER DE ANOTAÇÕES LABEL STUDIO
# ═══════════════════════════════════════════════════════════════════

def parse_annotation_results(results: List[Dict[str, Any]]) -> Tuple[
    Optional[Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    """
    Parseia a lista 'result' de um JSON de anotação do Label Studio.

    Retorna:
        bbox: dict com {x, y, width, height} normalizados [0-1]
              (cx, cy, w, h no sistema YOLO — convertidos de percentual para fração).
        keypoints: dict mapeando nome_kp -> {x, y, visibility}
                   onde x/y são frações [0-1] e visibility=2 (visível) ou 0 (oculto).
    """
    bbox: Optional[Dict[str, float]] = None
    keypoints: Dict[str, Dict[str, float]] = {}

    # Mapa de visibilidade por id de resultado
    visibility_map: Dict[str, str] = {}

    # Primeiro passo: coletar visibilidade
    for res in results:
        if res.get("from_name") == "visibility" and "choices" in res.get("value", {}):
            rid = res.get("id", "")
            choices = res["value"]["choices"]
            if choices:
                visibility_map[rid] = choices[0]

    # Segundo passo: extrair bbox e keypoints
    for res in results:
        val = res.get("value", {})

        # --- Bounding Box ---
        if "rectanglelabels" in val:
            # Label Studio usa porcentagens [0-100]
            rx = val.get("x", 0) / 100.0
            ry = val.get("y", 0) / 100.0
            rw = val.get("width", 0) / 100.0
            rh = val.get("height", 0) / 100.0
            # Converte para centro YOLO
            bbox = {
                "x": rx + rw / 2.0,
                "y": ry + rh / 2.0,
                "width": rw,
                "height": rh,
            }

        # --- Keypoints ---
        if "keypointlabels" in val:
            labels = val["keypointlabels"]
            if labels:
                kp_name = labels[0]
                kp_x = val.get("x", 0) / 100.0
                kp_y = val.get("y", 0) / 100.0

                # Determinar visibilidade
                rid = res.get("id", "")
                vis_label = visibility_map.get(rid, "Visível")
                vis_val = 2.0 if "vis" in vis_label.lower() else 0.0

                keypoints[kp_name] = {
                    "x": kp_x,
                    "y": kp_y,
                    "visibility": vis_val,
                }

    return bbox, keypoints


# ═══════════════════════════════════════════════════════════════════
#  VALIDAÇÃO DE ANOTAÇÃO
# ═══════════════════════════════════════════════════════════════════

def validate_annotation(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Valida uma lista de resultados de anotação.

    Retorna dict com:
        valid: bool
        has_bbox: bool
        missing_kps: list de nomes faltantes
        duplicate_kps: list de nomes duplicados
        found_kps: list de nomes encontrados
        issues: list de strings descrevendo problemas
    """
    has_bbox = False
    kp_counts: Dict[str, int] = {}
    issues: List[str] = []

    for res in results:
        val = res.get("value", {})

        if "rectanglelabels" in val:
            has_bbox = True

        if "keypointlabels" in val:
            labels = val.get("keypointlabels", [])
            if labels:
                name = labels[0]
                kp_counts[name] = kp_counts.get(name, 0) + 1

    # Verificar bbox
    if not has_bbox:
        issues.append("Bounding Box ausente")

    # Verificar keypoints faltantes
    found_kps = list(kp_counts.keys())
    missing_kps = [kp for kp in TARGET_KPS if kp not in kp_counts]
    if missing_kps:
        issues.append(f"Keypoints faltantes ({len(missing_kps)}): {', '.join(missing_kps)}")

    # Verificar duplicados
    duplicate_kps = [kp for kp, count in kp_counts.items() if count > 1]
    if duplicate_kps:
        issues.append(f"Keypoints duplicados: {', '.join(duplicate_kps)}")

    return {
        "valid": len(issues) == 0,
        "has_bbox": has_bbox,
        "missing_kps": missing_kps,
        "duplicate_kps": duplicate_kps,
        "found_kps": found_kps,
        "issues": issues,
    }


def load_annotation_json(json_path: Path) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Carrega um JSON de anotação do Label Studio e extrai:
        image_ref: string de referência da imagem (do campo task.data.img)
        results:   lista de resultados de anotação

    Suporta dois formatos:
        - JSON com 'result' no nível raiz (export de annotation)
        - JSON com 'annotations[0].result'
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extrair referência da imagem
    image_ref = None
    task = data.get("task", {})
    if isinstance(task, dict):
        task_data = task.get("data", {})
        image_ref = task_data.get("img") or task_data.get("image")

    # Se não encontrou em task, tenta no nível raiz
    if not image_ref:
        root_data = data.get("data", {})
        if isinstance(root_data, dict):
            image_ref = root_data.get("img") or root_data.get("image")

    # Extrair results
    results = data.get("result", [])
    if not results:
        annotations = data.get("annotations", [])
        if annotations and isinstance(annotations[0], dict):
            results = annotations[0].get("result", [])

    return image_ref, results
