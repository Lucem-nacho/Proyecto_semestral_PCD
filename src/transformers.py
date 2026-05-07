import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Imputación profesional: Reemplazamos espacios en blanco (incluso múltiples) por 0
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(r'^\s*$', '0', regex=True))

    # Mapeo del Target: Vital para que la IA pueda calcular probabilidades
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    return df