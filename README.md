# Telco Customer Churn - Data Preparation Pipeline

## Overview
Este repositorio contiene un pipeline profesional de ingeniería de datos diseñado para limpiar y transformar el dataset de **Telco Customer Churn**. El objetivo es identificar patrones de comportamiento de facturación y tipos de contrato que influyen en la fuga de clientes.

## Estructura del Proyecto
```text
proyecto_churn/
├── data/
│   ├── Raw/                  # Datasets originales (inmutables)
│   └── processed/            # Datos limpios y listos para ML
├── docs/                     # Rúbrica y Reporte Técnico
├── notebooks/                # Análisis Exploratorio (EDA) y experimentación de Pipelines
├── src/                      # Código fuente modular
│   ├── audit.py              # Validación de integridad
│   ├── transformers.py       # Limpieza y corrección de tipos
│   └── pipeline.py           # Transformaciones de preprocesamiento ML
├── main.py                   # Orquestador principal del flujo
├── requirements.txt          # Dependencias del proyecto
└── README.md                 # Instrucciones de uso