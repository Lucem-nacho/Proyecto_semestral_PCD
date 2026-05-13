"""
Custom data transformation functions for telecom churn dataset.
Includes functions for structural cleaning and target variable mapping.
"""

import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw DataFrame by handling missing values in specific columns
    and mapping the target variable to numeric format.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The raw input DataFrame.
        
    Returns
    -------
    pandas.DataFrame
        The cleaned DataFrame ready for pipeline processing.
    """
    df_copy = df.copy()
    
    # Imputación profesional: Reemplazamos espacios en blanco o celdas vacías por '0'
    # La regex '^\s+$|^$' asegura capturar espacios múltiples o cadenas vacías absolutas
    if 'TotalCharges' in df_copy.columns:
        df_copy['TotalCharges'] = pd.to_numeric(
            df_copy['TotalCharges'].replace(r'^\s+$|^$', '0', regex=True), 
            errors='coerce'
        ).fillna(0.0)

    # Mapeo del Target: Vital para que la IA (modelos de clasificación) entienda las clases
    if 'Churn' in df_copy.columns:
        df_copy['Churn'] = df_copy['Churn'].map({'Yes': 1, 'No': 0})
    
    # Eliminamos el identificador porque es ruido (no aporta valor predictivo)
    if 'customerID' in df_copy.columns:
        df_copy = df_copy.drop(columns=['customerID'])
        
    return df_copy