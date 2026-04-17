import pandas as pd
import joblib
import os

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS DINÁMICAS
# ==========================================
# Localizamos la carpeta donde está este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Subimos un nivel a la raíz del proyecto
ROOT_DIR = os.path.dirname(BASE_DIR)

# Definimos rutas absolutas a los archivos necesarios
PATH_MODEL = os.path.join(ROOT_DIR, 'models', 'modelo_churn_final.pkl')
PATH_COLUMNS = os.path.join(ROOT_DIR, 'models', 'columnas_modelo.pkl')
PATH_DATA_INPUT = os.path.join(ROOT_DIR, 'data', 'churn_raw.csv')
PATH_DATA_OUTPUT = os.path.join(ROOT_DIR, 'data', 'clientes_con_riesgo.csv')

def cargar_recursos():
    """Carga el modelo y las columnas necesarias."""
    if not os.path.exists(PATH_MODEL) or not os.path.exists(PATH_COLUMNS):
        raise FileNotFoundError(f"No se encontraron los modelos en {ROOT_DIR}/models. "
                                "Asegúrate de ejecutar primero el Notebook.")
    
    modelo = joblib.load(PATH_MODEL)
    columnas = joblib.load(PATH_COLUMNS)
    return modelo, columnas

def procesar_y_predecir(df_input, modelo, columnas_entrenamiento):
    """Limpia, transforma y predice sobre datos nuevos."""
    df = df_input.copy()
    
    # --- PREPROCESAMIENTO ---
    # Eliminamos identificadores (pero los mantenemos en el DF original para el resultado)
    df_proc = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')
    
    # Transformación de Género
    df_proc['Gender'] = df_proc['Gender'].map({'Female': 0, 'Male': 1})
    
    # One-Hot Encoding para Países (Geography)
    df_proc = pd.get_dummies(df_proc, columns=['Geography'], drop_first=True)
    
    # Alineación de columnas: Aseguramos que tenga las mismas columnas que vio el modelo
    for col in columnas_entrenamiento:
        if col not in df_proc.columns:
            df_proc[col] = 0
            
    # Ordenar columnas exactamente igual al entrenamiento
    df_proc = df_proc[columnas_entrenamiento]
    
    # --- PREDICCIÓN ---
    # Obtenemos la probabilidad (0.0 a 1.0)
    probabilidades = modelo.predict_proba(df_proc)[:, 1]
    
    # Añadimos resultados al dataframe original
    df['Probabilidad_Fuga'] = probabilidades
    df['Riesgo_Alto'] = (probabilidades > 0.5).astype(int)
    
    return df

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    try:
        print("🚀 Iniciando Pipeline de Predicción...")
        
        # 1. Cargar recursos
        model, cols = cargar_recursos()
        print("✅ Modelo y columnas cargados.")
        
        # 2. Cargar datos
        if not os.path.exists(PATH_DATA_INPUT):
            print(f"❌ Error: No se encuentra el archivo en {PATH_DATA_INPUT}")
        else:
            datos_nuevos = pd.read_csv(PATH_DATA_INPUT)
            
            # 3. Predecir
            resultado_final = procesar_y_predecir(datos_nuevos, model, cols)
            
            # 4. Guardar resultados
            resultado_final.to_csv(PATH_DATA_OUTPUT, index=False)
            print(f"✅ ¡Éxito! Archivo generado en: {PATH_DATA_OUTPUT}")
            print(f"📊 Se analizaron {len(resultado_final)} clientes.")

    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")