"""
Main orchestrator for the Telecom Churn ETL pipeline.
Executes auditing, cleaning, preprocessing, and exporting.

Usage:
    python main.py
"""

import pandas as pd
import traceback
from pathlib import Path

# Imports locales del proyecto
from src.audit import audit_data
from src.transformers import clean_data
from src.pipeline import build_preprocessing_pipeline

def main():
    """Executes the complete ETL pipeline."""
    print("="*60)
    print("📡 PIPELINE DE DATOS: ROTACIÓN DE CLIENTES (TELECOM CHURN)")
    print("="*60)
    
    try:
        # ============ 1. EXTRACCIÓN (CARGA DE DATOS) ============
        print("\n📥 Fase 1: Extracción de datos")
        # Ojo: Buscamos en la carpeta "Raw" con mayúscula, tal como la crearon
        data_dir = Path("data/Raw") 
        csv_files = list(data_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"❌ Error: No se encontró ningún archivo CSV en {data_dir}")
            return
            
        raw_path = csv_files[0]
        print(f"📁 Procesando archivo: {raw_path.name}")

        # ============ 2. AUDITORÍA INICIAL ============
        print("\n🔍 Fase 2: Auditoría de integridad")
        if not audit_data(raw_path): 
            print("❌ Error: El archivo no pasó la auditoría.")
            return

        # ============ 3. LIMPIEZA ESTRUCTURAL ============
        print("\n🧹 Fase 3: Limpieza estructural y de formato")
        df_raw = pd.read_csv(raw_path)
        df_cleaned = clean_data(df_raw)

        # ============ 4. PREPROCESAMIENTO (PIPELINE) ============
        print("\n🏗️  Fase 4: Construcción y aplicación del Pipeline")
        
        # Separamos 'X' e 'y' en el momento exacto (justo antes del entrenamiento)
        target_col = 'Churn'
        y = df_cleaned[target_col] if target_col in df_cleaned.columns else None
        X = df_cleaned.drop(columns=[target_col], errors='ignore')
        
        # Construimos el pipeline y transformamos SOLO las variables predictoras (X)
        pipeline = build_preprocessing_pipeline(X, target_col=target_col)
        X_processed = pipeline.fit_transform(X)
        
        # Reconstrucción de columnas (Limpieza de nombres de Scikit-Learn)
        try:
            feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
            feature_names = [n.split('__')[-1] for n in feature_names]
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
            
        df_final = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
        # Re-acoplamos la variable objetivo (y) limpia al final del dataset
        if y is not None:
            df_final[target_col] = y
        
        # ============ 5. CARGA (GUARDADO FINAL) ============
        print("\n💾 Fase 5: Guardado del dataset procesado")
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "processed_data.csv"

        df_final.to_csv(output_path, index=False)
        
        # ============ RESUMEN FINAL ============
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"   • Archivo generado:   {output_path}")
        print(f"   • Dimensiones finales: {df_final.shape}")
        print("\n✨ ¡Listo para el entrenamiento de Machine Learning!\n")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()