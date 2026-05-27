import pandas as pd
from pathlib import Path

def audit_data(file_path: str) -> bool:
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Error: Archivo no encontrado en: {file_path}")
        return False
        
    try:
        # 1. Leer solo los metadatos de las columnas (nrows=0) para máxima eficiencia
        df_check = pd.read_csv(path, nrows=0)
        
        # 2. Definir las columnas críticas que el pipeline necesita obligatoriamente
        required_columns = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'Churn']
        
        # 3. Validar la integridad del esquema
        missing_columns = [col for col in required_columns if col not in df_check.columns]
        if missing_columns:
            print(f"❌ Error de esquema: Faltan las siguientes columnas críticas: {missing_columns}")
            return False
            
        print("🔍 Auditoría estructural aprobada: El archivo existe y el esquema es válido.")
        return True
        
    except pd.errors.EmptyDataError:
        print(f"❌ Error de integridad: El archivo en {file_path} está completamente vacío.")
        return False
    except pd.errors.ParserError:
        print(f"❌ Error de formato: El archivo en {file_path} no es un CSV válido o está corrupto.")
        return False
    except Exception as e:
        print(f"❌ Error inesperado durante la fase de auditoría: {str(e)}")
        return False