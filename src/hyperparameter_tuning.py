import json
from pathlib import Path
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def optimize_hyperparameters(X_train, y_train, n_trials=8, cv=3) -> dict:
    """
    Ejecuta la búsqueda automatizada de hiperparámetros usando Optuna.
    Garantiza exactamente 24 entrenamientos (8 trials * 3 folds de Validación Cruzada).
    Exporta la combinación paramétrica ganadora en un archivo JSON independiente.
    """
    # Desactivar los mensajes repetitivos e interrupciones de Optuna en la consola
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        # 1. Optuna sugiere algoritmos lineal (LogisticRegression) o ensamblado (RandomForest)
        classifier_name = trial.suggest_categorical("classifier", ["RandomForest", "LogisticRegression"])
        
        if classifier_name == "RandomForest":
            # Espacio de búsqueda para Random Forest
            n_estimators = trial.suggest_int("rf_n_estimators", 50, 200, step=50)
            max_depth = trial.suggest_int("rf_max_depth", 5, 20)
            min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
            
        else:
            # Espacio de búsqueda para Regresión Logística
            C = trial.suggest_float("lr_C", 1e-4, 10.0, log=True)
            solver = trial.suggest_categorical("lr_solver", ["lbfgs", "liblinear"])
            
            model = LogisticRegression(
                C=C,
                solver=solver,
                max_iter=1000,
                random_state=42
            )
        
        # 2. Validación Cruzada optimizando F1-Score (Métrica clave por desbalance de clases)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        
        # Retornamos el promedio del F1-Score obtenido en los 3 folds
        return score.mean()

    print("🚀 Iniciando fase de optimización automatizada de hiperparámetros...")
    print(f"📊 Configuración: {n_trials} Ensayos × {cv} Folds = {n_trials * cv} entrenamientos reales calculados matemáticamente.")
    
    # Crear el estudio de Optuna maximizando la métrica objetivo
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"🏆 Optimización completada con éxito. Mejor F1-Score alcanzado en CV: {study.best_value:.4f}")
    print("🎯 Configuración ganadora guardada en memoria.")
    
    # 3. GUARDADO DE ARTEFACTO (Desacoplamiento de Ingeniería de producción)
    BASE_DIR = Path(__file__).resolve().parent.parent
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True) # Crea la carpeta si no existe
    config_path = models_dir / "best_hyperparameters.json"
    
    # Escribir los parámetros ganadores en un archivo JSON estructurado
    with open(config_path, "w") as f:
        json.dump(best_params, f, indent=4)
        
    print(f"💾 Artefacto JSON exportado exitosamente en: {config_path}")
    return best_params

# Bloque de prueba de sanidad e integración local
if __name__ == "__main__":
    from src.data_preprocessing import load_and_preprocess_data
    
    print("--- Probando Integración de Módulos Locales ---")
    # Consumir directamente los datos limpios y separados del módulo anterior sin Data Leakage
    X_train, _, y_train, _ = load_and_preprocess_data()
    
    # Ejecutar la optimización
    mejores_parametros = optimize_hyperparameters(X_train, y_train)