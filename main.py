import pandas as pd
from pathlib import Path

# IMPORTANTE: Estas líneas conectan tus archivos. 
# Si fallan, asegúrate de que tus archivos se llamen exactamente así en la carpeta /src
from src.audit import audit_data
from src.transformers import clean_data
from src.pipeline import build_preprocessing_pipeline

def main():
    try:
        # Búsqueda dinámica del archivo CSV
        data_dir = Path("data/Raw") # Ojo: en su repo la carpeta se llama "Raw" con mayúscula
        csv_files = list(data_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"Error: No se encontró ningún archivo CSV en {data_dir}")
            return
            
        raw_path = csv_files[0]
        print(f"Procesando archivo: {raw_path.name}")


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
        pipeline = build_preprocessing_pipeline(df_cleaned, target_col='Churn')
        
        y = df_cleaned['Churn']
        X = df_cleaned.drop(columns=['Churn'])
        
        # Transformamos TODO el dataframe (menos el ID)
        X_processed = pipeline.fit_transform(df_cleaned.drop(columns=['customerID'], errors='ignore'))
        
        # 4. Reconstrucción de columnas (Limpieza de nombres)
        feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
        feature_names = [n.split('__')[-1] for n in feature_names]
        df_final = pd.DataFrame(X_processed, columns=feature_names, index=df_cleaned.index)
        
        # 5. Guardado
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "processed_data.csv"

        df_final.to_csv(output_path, index=False)
        
        print("--- Pipeline finalizado exitosamente ---")
        print(f"Datos guardados en: {output_path}")
        print(f"Dimensiones finales: {df_final.shape[0]} filas x {df_final.shape[1]} columnas\n")

    except Exception as e:
        print(f"Error en la ejecución: {e}")

if __name__ == "__main__":
    main()