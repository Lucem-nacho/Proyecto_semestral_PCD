"""
Main orchestrator for the Telecom Churn Machine Learning Pipeline.
Executes the complete life cycle: Auditing, Anti-Leakage Preprocessing,
Hyperparameter Optimization (Optuna), Final Training, and Dynamic Evaluation.

Usage:
    python main.py
"""

import traceback
from pathlib import Path

# Imports de la arquitectura modular de soporte (Cero acoplamiento)
from src.audit import audit_data
from src.data_preprocessing import load_and_preprocess_data  # Ajustar a src.data_prep si cambiaste el nombre
from src.hyperparameter_tuning import optimize_hyperparameters
from src.model_training import train_final_model
from src.model_evaluation import evaluate_predictions

def main():
    """Orquesta el ciclo completo de Machine Learning sin intervención manual."""
    print("="*70)
    print("📡 SISTEMA MAESTRO DE IA: DETECCIÓN DE FUGA DE CLIENTES (TELCO CHURN)")
    print("="*70)
    
    try:
        # ============ 1. EXTRACCIÓN Y LOCALIZACIÓN ============
        print("\n📥 FASE 1: Extracción y localización de fuentes")
        data_dir = Path("data/Raw")
        if not data_dir.exists():
            data_dir = Path("data/raw")
            
        csv_files = list(data_dir.glob('*.csv'))
        if not csv_files:
            print(f"❌ Error crítico: No se encontró ningún archivo CSV en {data_dir}")
            return
            
        raw_path = csv_files[0]
        print(f"   • Archivo identificado para procesamiento: {raw_path.name}")

        # ============ 2. AUDITORÍA ESTRUCTURAL ============
        print("\n🔍 FASE 2: Auditoría automatizada de esquema e integridad")
        if not audit_data(raw_path): 
            print("❌ Error de integridad: El archivo no cumple con el contrato de datos mínimo.")
            return

        # ============ 3. PREPROCESAMIENTO ROBUSTO (ANTI-LEAKAGE) ============
        print("\n🏗️  FASE 3: Aislamiento de conjuntos y preprocesamiento diferenciado")
        # El módulo load_and_preprocess_data carga, limpia con transformers.py,
        # separa en Train/Test de forma estratificada y aplica las transformaciones de forma aislada.
        X_train, X_test, y_train, y_test = load_and_preprocess_data(test_size=0.2, random_state=42)
        print("   ✅ Matrices de datos consolidadas en memoria sin Data Leakage.")

        # ============ 4. OPTIMIZACIÓN DE HIPERPARAMETROS ============
        print("\n🚀 FASE 4: Ejecución del experimento paramétrico (Optuna)")
        # Llama a Optuna. Ejecuta 8 trials x 3 folds = 24 entrenamientos exactos. Exporta el JSON.
        mejores_params = optimize_hyperparameters(X_train, y_train, n_trials=8, cv=3)

        # ============ 5. ENTRENAMIENTO DEFINITIVO (PERSISTENCIA) ============
        print("\n🚂 FASE 5: Consolidación del modelo campeón en disco duro")
        # Lee el JSON generado en el paso anterior, entrena con el 100% de Train y guarda el archivo .joblib
        modelo_final = train_final_model(X_train, y_train)

        # ============ 6. EVALUACIÓN PRESCRIPTIVA EN TEST ============
        print("\n📊 FASE 6: Evaluación analítica final en el conjunto de control")
        # Ejecuta la evaluación aplicando el umbral optimizado de negocio (0.35) que propusimos
        resumen_metricas = evaluate_predictions(modelo_final, X_test, y_test, threshold=0.35)
        
        # ============ RESUMEN OPERATIVO FINAL ============
        print("\n" + "="*70)
        print("🏆 PIPELINE DE MACHINE LEARNING EJECUTADO DE EXTREMO A EXTREMO")
        print("="*70)
        print(f"   • Modelo Consolidado:       models/final_model.joblib")
        print(f"   • Configuración Guardada:    models/best_hyperparameters.json")
        print(f"   • Desempeño F1-Score Test:  {resumen_metricas['f1_churn']:.4f}")
        print(f"   • Capacidad AUC-ROC:        {resumen_metricas['auc_roc']:.4f}")
        print(f"   • Alarmas de Fuga Emitidas: {resumen_metricas['falsos_positivos'] + resumen_metricas['falsos_negativos'] + int(y_test.sum())}")
        print("="*70)
        print("✨ ¡Solución lista para la toma de decisiones estratégicas!\n")

    except Exception as e:
        print(f"\n❌ FATAL ERROR EN EL PROCESO ORQUESTRADOR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
    