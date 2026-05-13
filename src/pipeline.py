"""
Pipeline construction module for the Telecom Churn dataset.
Combines numerical scaling and categorical encoding.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline(df, target_col='Churn', extra_drop_cols=None):
    """
    Builds and returns a scikit-learn preprocessing pipeline.
    Automatically detects numeric features for scaling and categorical features for encoding.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe used to dynamically identify columns.
    target_col : str, default='Churn'
        The target variable to exclude from transformations.
    extra_drop_cols : list of str, optional
        Additional columns to drop (e.g., identifiers like 'customerID').
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        The assembled preprocessing pipeline.
    """
    if extra_drop_cols is None:
        # Se excluye el identificador único por defecto para evitar sobreajuste
        extra_drop_cols = ['customerID']
        
    # Definición de columnas a excluir del entrenamiento
    cols_to_exclude = [target_col] + extra_drop_cols
    feature_df = df.drop(columns=[col for col in cols_to_exclude if col in df.columns], errors='ignore')
    
    # Identificación dinámica de tipos de variables
    numeric_features = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Transformador para variables numéricas (Estandarización)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 2. Transformador para variables categóricas (One-Hot Encoding)
    # drop='first' ayuda a evitar la multicolinealidad perfecta (Dummy Variable Trap)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # 3. Ensamblaje en ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
        
    # 4. Envoltorio final
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor)
    ])
    
    return pipeline