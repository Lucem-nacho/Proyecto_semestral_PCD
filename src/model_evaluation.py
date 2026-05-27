import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_predictions(model, X_test, y_test, threshold=0.5) -> dict:
    """
    Ejecuta una auditoría de rendimiento exhaustiva sobre el conjunto de Test.
    Permite ajustar dinámicamente el umbral de decisión probabilístico para optimizar 
    el Recall frente a los Falsos Negativos del negocio.
    Retorna un diccionario con las métricas clave para su registro en el pipeline.
    """
    print(f"\n📊 Iniciando evaluación definitiva del modelo (Umbral de decisión: {threshold})...")
    
    # 1. Extraer las probabilidades continuas de la clase positiva (Churn = 1)
    # Esto mide la certeza matemática que tiene el modelo antes de clasificar
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # 2. Aplicar el umbral dinámico para generar las predicciones binarias finales
    y_pred = (y_probs >= threshold).astype(int)
    
    # 3. Calcular métricas estándar de clasificación
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    
    print("\n=== REPORTE DE CLASIFICACIÓN (MÉTRICAS DE CONTROL) ===")
    print(report_text)
    
    # 4. Calcular la Matriz de Confusión Financiera
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("=== MATRIZ DE CONFUSIÓN OPERATIVA ===")
    print(f"   • Verdaderos Negativos (Clientes estables detectados): {tn}")
    print(f"   • Falsos Positivos (Alertas erróneas / Gasto de marketing): {fp}")
    print(f"   • Falsos Negativos (🚨 FUGAS NO DETECTADAS / PÉRDIDA DIRECTA): {fn}")
    print(f"   • Verdaderos Positivos (Clientes retenidos con éxito): {tp}\n")
    
    # 5. Calcular la métrica AUC-ROC (Capacidad absoluta de discriminación)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    print(f"📈 Área Bajo la Curva ROC (AUC-ROC): {roc_auc:.4f}")
    
    # 6. Consolidar métricas clave en un diccionario para la automatización del pipeline
    metrics_summary = {
        "accuracy": report_dict["accuracy"],
        "precision_churn": report_dict["1"]["precision"],
        "recall_churn": report_dict["1"]["recall"],
        "f1_churn": report_dict["1"]["f1-score"],
        "auc_roc": roc_auc,
        "falsos_negativos": int(fn),
        "falsos_positivos": int(fp)
    }
    
    return metrics_summary

# Bloque de validación local de integración y sanidad estructural
if __name__ == "__main__":
    from src.data_preprocessing import load_and_preprocess_data
    from sklearn.linear_model import LogisticRegression
    
    print("--- Probando Integración del Módulo de Evaluación ---")
    
    # Cargar datos limpios sin Data Leakage
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Instanciar un modelo base rápido para verificar que el evaluador no falle
    mock_model = LogisticRegression(max_iter=1000, random_state=42)
    mock_model.fit(X_train, y_train)
    
    # Evaluar simulando el umbral optimizado de negocio (0.35) que propusimos en las conclusiones
    resumen_metricas = evaluate_predictions(mock_model, X_test, y_test, threshold=0.35)
    print("\n✅ Módulo de evaluación completado y validado de manera aislada.")