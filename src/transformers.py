"""
Custom data transformation functions for telecom churn dataset.
Includes functions for structural cleaning and target variable mapping.
"""

import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame crudo resolviendo inconsistencias de tipos de datos,
    removiendo ruido estructural y codificando la variable objetivo de forma idempotente.
    
    Parameters
    ----------
    df : pandas.DataFrame
        El DataFrame crudo de entrada.
        
    Returns
    -------
    pandas.DataFrame
        El DataFrame limpio y estandarizado listo para el pipeline de características.
    """
    df_copy = df.copy()
    
    # 1. Imputación profesional e ingeniería de tipos para la columna defectuosa
    if 'TotalCharges' in df_copy.columns:
        # La regex captura de forma segura espacios múltiples o celdas vacías absolutas
        df_copy['TotalCharges'] = df_copy['TotalCharges'].replace(r'^\s+$|^$', '0', regex=True)
        df_copy['TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'], errors='coerce').fillna(0.0)

    # 2. Mapeo Idempotente del Target y Forzado de Tipo Numérico Puro
    # CORRECCIÓN CRÍTICA: Se añade pd.to_numeric y .astype(int) para asegurar que scikit-learn
    # no reciba un tipo 'object' o mixto, solucionando el error 'Unknown label type: unknown'.
    if 'Churn' in df_copy.columns:
        df_copy['Churn'] = df_copy['Churn'].replace({'Yes': 1, 'No': 0})
        df_copy['Churn'] = pd.to_numeric(df_copy['Churn'], errors='coerce').fillna(0).astype(int)
    
    # 3. Eliminación de ruido estructural (Identificadores sin poder predictivo)
    if 'customerID' in df_copy.columns:
        df_copy = df_copy.drop(columns=['customerID'])
        
    return df_copy