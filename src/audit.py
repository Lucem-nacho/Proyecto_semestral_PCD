import pandas as pd
from pathlib import Path

def audit_data(file_path: str) -> bool:
    path = Path(file_path)
    if not path.exists():
        print(f"Archivo no encontrado en: {file_path}")
        return False
        
    # Verificación de integridad básica
    df_check = pd.read_csv(file_path, nrows=1)
    if df_check.empty:
        return False
        
    return True