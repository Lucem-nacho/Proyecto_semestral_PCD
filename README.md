# Telco Customer Churn - End-to-End Machine Learning Pipeline

## Overview
Este repositorio contiene un flujo de trabajo profesional de Ciencia de Datos orientado a predecir y mitigar la fuga de clientes (Churn) en una empresa de telecomunicaciones. El proyecto abarca desde la ingeniería de datos inicial hasta la implementación y optimización de modelos de Machine Learning (Supervisados y No Supervisados). 

El objetivo principal es identificar patrones de riesgo mediante algoritmos predictivos y segmentación espacial, proporcionando insights accionables para la retención estratégica de clientes.

## Estructura del Proyecto
```text
proyecto_churn/
├── data/
│   ├── Raw/                  # Datasets originales (inmutables)
│   └── processed/            # Datos limpios y estandarizados tras el pipeline
├── docs/                     # Rúbricas, Informes Word/PDF y Reporte Técnico
├── models/                   # Artefactos exportados
│   ├── best_hyperparameters.json # Configuración óptima generada por Optuna
│   └── final_model.joblib    # Modelo de Regresión Logística entrenado
├── notebooks/                # Cuadernos Jupyter del ciclo de vida ML
│   ├── 01_exploratory_analisis.ipynb
│   ├── 02_supervised_modeling.ipynb
│   ├── 03_unsupervised_modeling.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   └── 05_final_analysis.ipynb
├── src/                      # Código fuente modular (Pipeline MLOps)
│   ├── audit.py              # Validación de integridad de datos
│   ├── data_preprocessing.py # Lógica de división y limpieza
│   ├── pipeline.py           # Transformaciones (StandardScaler, OneHotEncoder)
│   ├── transformers.py       # Corrección de tipos y curación
│   ├── model_training.py     # Lógica de entrenamiento algorítmico
│   ├── hyperparameter_tuning.py # Búsqueda automatizada con Optuna
│   └── model_evaluation.py   # Métricas (AUC, ROC, F1-Score)
├── main.py                   # Orquestador principal del flujo
├── requirements.txt          # Dependencias del proyecto (Scikit-Learn, Optuna, etc.)
└── README.md                 # Documentación del repositorio