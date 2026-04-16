import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Imputación profesional: No borramos, transformamos a 0 los clientes nuevos
    df['TotalCharges'] = df['TotalCharges'].replace(" ", "0")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Mapeo del Target: Vital para que la IA pueda calcular probabilidades
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    return df