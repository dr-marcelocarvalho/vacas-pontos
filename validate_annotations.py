
import json
from pathlib import Path

# Configuração
KEYPOINTS_REQ = [
    "withers", "back", "hook up", "hook down", "hip", "tail head", "pin up", "pin down"
]

BASE_DIR = Path("g:/PYTHON/cow")
JSON_DIR = BASE_DIR / "data/jsons_ls"

def validate_annotations():
    print(f"Validando anotações em: {JSON_DIR}")
    json_files = list(JSON_DIR.glob("*.json"))
    
    if not json_files:
        json_files = list(JSON_DIR.glob("*"))

    items_ok = 0
    items_error = 0
    
    print(f"Total de arquivos encontrados: {len(json_files)}\n")

    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            # Extrair Metadados
            task_id = data.get("id") or data.get("task", {}).get("id")
            
            username = "N/A"
            email = "N/A"
            
            annotations = data.get("annotations", [])
            if annotations:
                ann = annotations[0]
                username = ann.get("created_username", "N/A")
                email = ann.get("created_email", "N/A")
                if username == "N/A" and "completed_by" in ann:
                    cb = ann.get("completed_by", {})
                    if isinstance(cb, dict):
                        email = cb.get("email", "N/A")
                        first = cb.get("first_name", "")
                        last = cb.get("last_name", "")
                        if first or last:
                            username = f"{first} {last}".strip()

            elif "created_username" in data:
                 username = data.get("created_username", "N/A")
                 email = data.get("created_email", "N/A")

            results = data.get("result", [])
            if not results and annotations:
                 results = annotations[0].get("result", [])

            has_bbox = False
            found_kps = set()
            
            for res in results:
                val = res.get("value", {})
                
                if "rectanglelabels" in val:
                    has_bbox = True
                
                if "keypointlabels" in val:
                    lbls = val.get("keypointlabels", [])
                    if lbls:
                        found_kps.add(lbls[0])
            
            # Verificação
            missing_kps = [k for k in KEYPOINTS_REQ if k not in found_kps]
            
            issues = []
            if not has_bbox:
                issues.append("BBox Faltando")
            if missing_kps:
                issues.append(f"Keypoints Faltando ({len(missing_kps)}): {', '.join(missing_kps)}")
            
            if issues:
                print(f"[ERRO] {jf.name}:")
                print(f"  Task ID: {task_id} | User: {username} | Email: {email}")
                for i in issues:
                    print(f"  - {i}")
                items_error += 1
            else:
                items_ok += 1
                
        except Exception as e:
            print(f"[FALHA AO LER] {jf.name}: {e}")
            items_error += 1

    print("\n" + "="*30)
    print("RELATÓRIO FINAL")
    print(f"Arquivos OK: {items_ok}")
    print(f"Arquivos com Erro: {items_error}")
    print("="*30)

if __name__ == "__main__":
    validate_annotations()
