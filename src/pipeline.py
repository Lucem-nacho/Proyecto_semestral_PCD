import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline(df, target_col='Churn', extra_drop_cols=None):
    """
    Construye y retorna un pipeline de preprocesamiento de Scikit-Learn.
    Detecta automaticamente variables numericas para escalado y categoricas para codificacion.
    """
    if extra_drop_cols is None:
        # Se excluye el identificador unico por defecto para evitar sobreajuste
        extra_drop_cols = ['customerID']
        
    # Definicion de columnas a excluir del entrenamiento
    cols_to_exclude = [target_col] + extra_drop_cols
    feature_df = df.drop(columns=[col for col in cols_to_exclude if col in df.columns])
    
    # Identificacion de tipos de variables
    numeric_features = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Transformador para variables numericas
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 2. Transformador para variables categoricas
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # 3. Ensamblaje en ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    # 4. Envoltorio final
    # Se nombra el paso como 'preprocessing' para mantener consistencia con la extraccion de nombres
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor)
    ])
    
    return pipeline