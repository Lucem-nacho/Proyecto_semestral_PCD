# Reporte Técnico: Predicción de Fuga de Clientes (Telco)

## 1. Introducción
Este informe detalla el proceso de ingeniería de datos y análisis exploratorio realizado sobre el dataset de Telco Customer Churn. El objetivo es proporcionar una base de datos limpia y preprocesada para modelos de Machine Learning.

## 2. Pregunta de Negocio
¿Podemos predecir qué clientes tienen más probabilidad de abandonar la empresa basándonos en su comportamiento de facturación y tipo de contrato, antes de que sea demasiado tarde?

## 3. Metodología
Se implementó un pipeline de datos modular que consta de:
* **Auditoría de Datos:** Verificación de integridad, tipos de datos y presencia de columnas críticas.
* **Limpieza Técnica:** Conversión de la variable `TotalCharges` a tipo numérico y gestión de valores nulos (11 registros eliminados por falta de historial de facturación).
* **Preprocesamiento:** Escalado de variables numéricas (`StandardScaler`) y codificación de variables categóricas (`OneHotEncoder`).

## 4. Hallazgos Clave
* **Impacto del Contrato:** Los clientes con contratos mes a mes presentan la tasa más alta de abandono.
* **Sensibilidad al Precio:** Existe una correlación visual clara entre cargos mensuales elevados y la decisión de abandonar el servicio.
* **Integridad:** El 99.8% de los datos son consistentes, lo que permite un modelado confiable.

## 5. Conclusión
El pipeline desarrollado garantiza que los datos están listos para la fase de modelado predictivo, cumpliendo con los estándares de organización y reproducibilidad exigidos.

---

### 2. Actualización para `docs/technical_report.md`

Este reporte ahora incluye los resultados algorítmicos (Accuracy, F1-Score, AUC), el análisis de clústeres y las recomendaciones de negocio. 

# Reporte Técnico: Predicción y Segmentación de Fuga de Clientes (Telco)

## 1. Introducción
Este informe detalla el ciclo completo de desarrollo de una solución de Machine Learning aplicada al dataset de Telco Customer Churn. El proyecto transitó desde la ingeniería y limpieza de datos hasta el modelado avanzado, implementando técnicas de aprendizaje supervisado, no supervisado y optimización paramétrica para resolver un problema crítico de negocio.

## 2. Pregunta de Negocio
¿Podemos predecir con precisión qué clientes tienen la mayor probabilidad de abandonar la empresa y, simultáneamente, agruparlos según sus perfiles de consumo para diseñar estrategias de retención personalizadas y rentables?

## 3. Metodología y Pipeline
Se implementó una arquitectura modular que previene la fuga de datos (Data Leakage) dividiendo el conjunto en Entrenamiento (80%) y Prueba (20%) antes de cualquier transformación. El flujo consistió en:
* **Ingeniería de Datos:** Imputación de nulos, transformación de tipos (ej. `TotalCharges`) y aplicación de `StandardScaler` y `OneHotEncoder`.
* **Modelado Supervisado:** Evaluación competitiva entre algoritmos (Random Forest y Regresión Logística). La sintonización fina de hiperparámetros se automatizó utilizando **Optuna**, maximizando el F1-Score mediante validación cruzada.
* **Modelado No Supervisado:** Reducción de dimensionalidad con PCA y segmentación de clientes mediante **K-Means**, validando el número óptimo de clústeres mediante el Método del Codo y el Coeficiente de Silueta.

## 4. Resultados Algorítmicos
* **El Modelo Predictivo:** La **Regresión Logística** superó a los modelos basados en árboles, logrando una Exactitud (Accuracy) del 80% y un Área Bajo la Curva (**AUC ROC**) de 0.84 en datos no vistos.
* **Manejo del Desbalance:** Dado que la fuga es un evento minoritario (27%), se optimizó el F1-Score (0.59), balanceando un Precision del 65% (minimizando falsas alarmas financieras) y un Recall del 54% (capturando a más de la mitad de los desertores reales).
* **Segmentación Estructural:** K-Means reveló 3 nichos de negocio:
  * **Cluster 0:** Nuevos clientes en fase de exploración.
  * **Cluster 1:** Base altamente fidelizada, contratos a largo plazo y rentabilidad estable (Churn cercano al 0%).
  * **Cluster 2 (Zona Crítica):** Clientes caracterizados por contratos flexibles "Mes a Mes" y cargos mensuales elevados (principalmente servicios de Fibra Óptica). Este grupo concentra casi la totalidad del riesgo de abandono.

## 5. Conclusión y Recomendación Comercial
El sistema desarrollado demuestra que el abandono responde a factores estructurales matemáticamente predecibles, con el **Cluster 2** siendo el foco crítico de vulnerabilidad. 

Se recomienda a las gerencias no utilizar las predicciones de forma binaria, sino aplicar las probabilidades arrojadas por la Regresión Logística para identificar umbrales de riesgo (ej. riesgo > 70%). Las campañas de retención deben dirigir sus presupuestos a ofrecer subsidios estratégicos temporales a los clientes del Cluster 2, incentivando la transición de sus costosos contratos "Mes a Mes" hacia formatos "Anuales", erradicando así la raíz del Churn y protegiendo el ciclo de vida del usuario.