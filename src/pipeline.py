"""
Pipeline construction module for the Telecom Churn dataset.
Combines numerical scaling and categorical encoding cleanly without Data Leakage.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline(X_train) -> ColumnTransformer:
    """
    Construye y retorna un ColumnTransformer de Scikit-Learn optimizado.
    Identifica dinámicamente las columnas numéricas y categóricas basándose 
    exclusivamente en la matriz de características de entrenamiento (X_train).
    
    Parameters
    ----------
    X_train : pandas.DataFrame
        La matriz de características de entrenamiento (sin la variable objetivo 'Churn' 
        ni identificadores como 'customerID').
        
    Returns
    -------
    sklearn.compose.ColumnTransformer
        El transformador de columnas listo para aplicar .fit_transform()
    """
    # 1. Identificación dinámica de tipos de variables basada únicamente en la matriz X
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Transformador para variables numéricas (Estandarización)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 3. Transformador para variables categóricas (One-Hot Encoding)
    # CORRECCIÓN CRÍTICA: Se cambia handle_unknown a 'error' para convivir con drop='first'.
    # drop='first' es obligatorio para mitigar la trampa de la variable dummy en modelos lineales.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='error'))
    ])
    
    # 4. Ensamblaje definitivo en ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Mantiene intactas columnas binarias pre-calculadas si existiesen
    )
    
    print(f"⚙️ Pipeline estructurado con éxito:")
    print(f"   • {len(numeric_features)} Características Numéricas escaladas.")
    print(f"   • {len(categorical_features)} Características Categóricas codificadas (drop='first').")
        
    return preprocessor