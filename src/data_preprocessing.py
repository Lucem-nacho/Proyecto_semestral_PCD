import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_preprocess_data():
    """
    Carga el dataset crudo, aplica limpieza de nulos ocultos y 
    devuelve la matriz procesada (X) y el vector objetivo (y).
    """
    # 1. Rutas dinámicas
    BASE_DIR = Path(__file__).resolve().parent.parent
    raw_path = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Carga de datos
    df = pd.read_csv(raw_path)
    
    # 2. Limpieza e Imputación Profesional
    df['TotalCharges'] = df['TotalCharges'].replace(" ", "0")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Mapeo temprano de la variable objetivo
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Eliminación de columnas sin valor predictivo
    df = df.drop(columns=['customerID'])
    
    # Separación de características y variable objetivo
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # 3. Pipeline de Preprocesamiento (Scikit-Learn)
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Transformación final y reconstrucción del DataFrame
    X_processed = preprocessor.fit_transform(X)
    feature_names = [n.split('__')[-1] for n in preprocessor.get_feature_names_out()]
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    return X_df, y

# Bloque de prueba local
if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    print(f"Dataset procesado con éxito. Dimensiones de X: {X.shape}")