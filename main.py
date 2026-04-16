import pandas as pd
from pathlib import Path

# IMPORTANTE: Estas líneas conectan tus archivos. 
# Si fallan, asegúrate de que tus archivos se llamen exactamente así en la carpeta /src
from src.audit import audit_data
from src.transformers import clean_data
from src.pipeline import build_preprocessing_pipeline

def main():
    try:
        raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        
        # 1. Auditoría
        if not audit_data(raw_path): 
            print("Error: El archivo no pasó la auditoría.")
            return

        # 2. Carga y Limpieza
        df_raw = pd.read_csv(raw_path)
        # Corregido: Solo pasamos df_raw para evitar el TypeError anterior
        df_cleaned = clean_data(df_raw)

        # 3. Pipeline de Entrenamiento (IA)
        # Argumento: Pasamos los datos limpios para que la IA aprenda los rangos correctos
        pipeline = build_preprocessing_pipeline(df_cleaned)
        
        y = df_cleaned['Churn']
        X = df_cleaned.drop(columns=['Churn'])
        
        X_processed = pipeline.fit_transform(X)
        
        # 4. Reconstrucción de columnas (Limpieza de nombres)
        feature_names = [n.split('__')[-1] for n in pipeline.named_steps['preprocessing'].get_feature_names_out()]
        df_final = pd.DataFrame(X_processed, columns=feature_names)
        df_final['Churn'] = y.values
        
        # 5. Guardado
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_dir / "processed_data.csv", index=False)
        
        print("--- Pipeline finalizado exitosamente ---")

    except Exception as e:
        print(f"Error en la ejecución: {e}")

if __name__ == "__main__":
    main()