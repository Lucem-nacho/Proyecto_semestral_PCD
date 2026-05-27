import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_final_model(X_train, y_train) -> object:
    """
    Carga los hiperparámetros óptimos desde el artefacto JSON,
    instancia el algoritmo ganador, ejecuta el entrenamiento definitivo
    y serializa el modelo entrenado en un archivo físico (.joblib).
    """
    print("\n🚂 Iniciando fase de entrenamiento definitivo del modelo campeón...")
    
    # 1. Determinar rutas base del proyecto y localizar el archivo de configuración
    BASE_DIR = Path(__file__).resolve().parent.parent
    config_path = BASE_DIR / "models" / "best_hyperparameters.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"❌ Error crítico: No se encontró el archivo de hiperparámetros en {config_path}. "
            "Debes ejecutar primero la fase de optimización (hyperparameter_tuning.py)."
        )
        
    # 2. Leer la configuración óptima pre-calculada
    with open(config_path, "r") as f:
        best_params = json.load(f)
        
    best_model_name = best_params.get("classifier")
    print(f"🎯 Configuración recuperada. Modelo ganador detectado: {best_model_name}")
    
    # 3. Instanciación dinámica del modelo con sus parámetros optimizados
    if best_model_name == "RandomForest":
        final_model = RandomForestClassifier(
            n_estimators=best_params["rf_n_estimators"],
            max_depth=best_params["rf_max_depth"],
            min_samples_split=best_params["rf_min_samples_split"],
            random_state=42,
            n_jobs=-1
        )
    elif best_model_name == "LogisticRegression":
        final_model = LogisticRegression(
            C=best_params["lr_C"],
            solver=best_params["lr_solver"],
            max_iter=1000,
            random_state=42
        )
    else:
        raise ValueError(f"❌ Algoritmo no soportado o desconocido: {best_model_name}")
        
    # 4. Ajustar el modelo utilizando el 100% de los datos de entrenamiento preparados
    print("⏳ Ajustando parámetros matemáticos del clasificador (fit)...")
    final_model.fit(X_train, y_train)
    print("✅ Entrenamiento completado de manera exitosa.")
    
    # 5. SERIALIZACIÓN (Persistencia del modelo en disco duro - Estándar MLOps)
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_export_path = models_dir / "final_model.joblib"
    
    # Guardar el objeto del modelo entrenado
    joblib.dump(final_model, model_export_path)
    print(f"💾 ¡Modelo entrenado exportado físicamente en: {model_export_path}!")
    
    return final_model

# Bloque de validación local de integración y sanidad estructural
if __name__ == "__main__":
    from src.data_preprocessing import load_and_preprocess_data
    
    print("--- Probando Integración del Módulo de Entrenamiento ---")
    try:
        # Cargar datos limpios sin Data Leakage desde tu módulo de preprocesamiento
        X_train, _, y_train, _ = load_and_preprocess_data()
        
        # Ejecutar el flujo de entrenamiento definitivo
        modelo_objeto = train_final_model(X_train, y_train)
        print("\n✅ Módulo de entrenamiento validado de manera aislada y listo para producción.")
        
    except Exception as e:
        print(f"\n❌ Error durante la prueba local: {str(e)}")
        