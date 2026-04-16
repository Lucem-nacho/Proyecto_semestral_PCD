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