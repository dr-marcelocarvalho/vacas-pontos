"""
validate.py — Validação das Anotações do Label Studio.

Itera sobre todos os JSONs em Key_points/, verifica se cada anotação
contém BBox e os 8 keypoints obrigatórios, e gera um relatório
em outputs/reports/validation_report.json.
"""

import json
import sys
from pathlib import Path

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core_utils import (
    KEYPOINTS_DIR,
    OUTPUTS_DIR,
    load_annotation_json,
    validate_annotation,
)


def main() -> None:
    reports_dir = OUTPUTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(KEYPOINTS_DIR.glob("*"))
    json_files = [f for f in json_files if f.is_file() and f.name != ".DS_Store"]

    print("=" * 60)
    print("  🐄 VALIDAÇÃO DE ANOTAÇÕES — Cows Challenge")
    print("=" * 60)
    print(f"  Diretório: {KEYPOINTS_DIR}")
    print(f"  Total de arquivos: {len(json_files)}")
    print("=" * 60)

    report: dict = {
        "total": len(json_files),
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "details": [],
    }

    for jf in json_files:
        try:
            image_ref, results = load_annotation_json(jf)
            validation = validate_annotation(results)

            entry = {
                "file": jf.name,
                "image_ref": image_ref,
                **validation,
            }
            report["details"].append(entry)

            if validation["valid"]:
                report["valid"] += 1
            else:
                report["invalid"] += 1
                report["errors"].append({
                    "file": jf.name,
                    "issues": validation["issues"],
                })
                print(f"  ❌ {jf.name}")
                for issue in validation["issues"]:
                    print(f"     └─ {issue}")

        except Exception as e:
            report["invalid"] += 1
            report["errors"].append({
                "file": jf.name,
                "issues": [f"Erro ao ler: {str(e)}"],
            })
            print(f"  💥 {jf.name}: {e}")

    # Resumo
    print()
    print("=" * 60)
    print("  RELATÓRIO FINAL")
    print(f"  ✅ Válidos:   {report['valid']}")
    print(f"  ❌ Inválidos: {report['invalid']}")
    print(f"  📊 Total:     {report['total']}")
    print("=" * 60)

    # Salvar relatório
    output_path = reports_dir / "validation_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  📁 Relatório salvo em: {output_path}")


if __name__ == "__main__":
    main()
