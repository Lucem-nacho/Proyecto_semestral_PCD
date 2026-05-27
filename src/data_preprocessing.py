import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# IMPORTACIÓN MÁNDATORIA: Reutilizamos la única fuente de verdad de limpieza
from src.transformers import clean_data

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Carga el dataset crudo, delega la limpieza al módulo de transformers,
    divide en Train/Test de forma estratificada y aplica transformaciones aisladas.
    """
    # 1. Resolución de rutas dinámicas
    BASE_DIR = Path(__file__).resolve().parent.parent
    raw_path = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not raw_path.exists():
        raw_path = BASE_DIR / "data" / "Raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    df_raw = pd.read_csv(raw_path)
    
    # 2. DELEGACIÓN DE LIMPIEZA AUTOMATIZADA (Cero duplicación de código)
    df = clean_data(df_raw)
    
    # Separación de características y variable objetivo
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # 3. DIVISIÓN TRAIN/TEST ESTRATIFICADA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    # 4. Pipeline de Preprocesamiento (Scikit-Learn)
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    from src.pipeline import build_preprocessing_pipeline
    preprocessor = build_preprocessing_pipeline(X_train)
    
    # 5. Transformación Diferenciada Estricta anti-leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 6. Reconstrucción de los DataFrames y Alineación de Índices
    feature_names = [n.split('__')[-1] for n in preprocessor.get_feature_names_out()]
    
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names).reset_index(drop=True)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names).reset_index(drop=True)
    
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train_df, X_test_df, y_train, y_test