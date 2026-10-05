"""La misma Pipeline ajustada transforma entrenamiento e inferencia."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
CATEGORICAL = ['Geography','Gender']
FEATURES = NUMERIC + CATEGORICAL

def features(data):
    missing = set(FEATURES)-set(data.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    result = data[FEATURES].copy()
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col],errors='raise')
        if not np.isfinite(result[col]).all():
            raise ValueError(f'{col} debe contener números finitos')
    for col in CATEGORICAL:
        if result[col].isna().any() or result[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{col} necesita valores no vacíos')
    return result

def make_model():
    return Pipeline([
        ('preprocess',ColumnTransformer([('countries',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL),('numeric','passthrough',NUMERIC)])),
        ('classifier',RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)),
    ])

def predict(data, model):
    result = data.drop(columns=['Exited'],errors='ignore').copy()
    result['Probabilidad_Fuga'] = model.predict_proba(features(data))[:,list(model.classes_).index(1)]
    result['Riesgo_Alto'] = (result.Probabilidad_Fuga>=0.5).astype(int)
    return result
