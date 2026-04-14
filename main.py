import pandas as pd
from pathlib import Path
from src.audit import audit_data
from src.transformers import clean_data
from src.pipeline import build_preprocessing_pipeline

def main():
    """
    Script principal de orquestacion para el proyecto de prediccion de rotacion.
    Coordina la auditoria, carga, limpieza, transformacion y guardado de datos.
    """
    print("--- Iniciando Pipeline de Preparacion de Datos: Telco Churn ---")

    try:
        # 1. Fase de Auditoria
        raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if not audit_data(raw_path):
            print("Pipeline detenido: El archivo no supero las validaciones de auditoria.")
            return

        # 2. Carga de Datos
        print(f"\nCargando datos desde: {raw_path}")
        df_raw = pd.read_csv(raw_path)

        # 3. Limpieza Tecnica
        print("Aplicando limpieza inicial...")
        df_cleaned = clean_data(df_raw)

        # 4. Transformacion y Pipeline ML
        print("Construyendo y aplicando pipeline de preprocesamiento...")
        pipeline = build_preprocessing_pipeline(df_cleaned)
        
        # Separacion de la variable objetivo para la transformacion
        y = df_cleaned['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        X_processed = pipeline.fit_transform(df_cleaned)
        
        # Extraccion de los nombres de las columnas despues del OneHotEncoding
        feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
        
        # Limpieza visual de los prefijos generados por ColumnTransformer
        feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
        
        # Reconstruccion del DataFrame final
        df_processed = pd.DataFrame(X_processed, columns=feature_names)
        df_processed['Churn'] = y.values
        
        # 5. Fase de Guardado
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / "processed_data.csv"
        
        df_processed.to_csv(output_path, index=False)
        print(f"\nResultado: El dataset procesado se ha guardado en {output_path}")
        print(f"Dimensiones finales: {df_processed.shape}")
        print("--- Pipeline finalizado exitosamente ---")

    except FileNotFoundError:
        print("Error critico: No se encontro el archivo en la ruta especificada.")
    except pd.errors.EmptyDataError:
        print("Error critico: El archivo CSV esta vacio.")
    except Exception as e:
        print(f"Error fatal: El pipeline fallo inesperadamente debido a: {e}")

if __name__ == "__main__":
    main()