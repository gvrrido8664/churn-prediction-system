🛡️ Sistema de Predicción de Fuga de Clientes (Churn) - Machine Learning
Este proyecto desarrolla una solución integral de Analítica Predictiva para identificar clientes con alta probabilidad de abandonar una institución financiera (aplicable a Banca y Seguros como SURA). El sistema combina un modelo de Inteligencia Artificial en Python con un Dashboard interactivo en Power BI para la toma de decisiones estratégicas.

📊 Impacto del Proyecto
Precisión del Modelo: 86.60% utilizando el algoritmo Random Forest.

Variable Crítica Identificada: La Edad resultó ser el predictor más fuerte de fuga.

Valor de Negocio: Permite al equipo comercial priorizar la retención de clientes según el "Score de Riesgo" y el saldo en cuenta (Balance).

🏗️ Estructura del Proyecto
Plaintext
├── data/               # Datasets originales y procesados por la IA.
├── models/             # El modelo entrenado (.pkl) y metadatos.
├── notebooks/          # Análisis exploratorio (EDA) y entrenamiento.
├── scripts/            # Script de inferencia para datos nuevos.
└── dashboard/          # Archivo .pbix con la visualización final.
🛠️ Stack Tecnológico
Python 3.x: (Pandas, Scikit-Learn, Seaborn, Joblib).

Machine Learning: Random Forest Classifier (Bosque Aleatorio).

Power BI: Modelado de datos, medidas DAX y visualización predictiva.

GitHub: Control de versiones y gestión de directorios.

🚀 Cómo ejecutar el Pipeline
Entrenamiento: El análisis y entrenamiento se encuentra en notebooks/01_eda_limpieza.ipynb.

Generación de Predicciones: Ejecuta el script de Python para procesar clientes actuales:

python scripts/predict_churn.py

Visualización: Abre dashboard/Analisis_Predictivo_Fuga_Clientes.pbix y actualiza el origen de datos con el archivo clientes_con_riesgo.csv generado.

🔍 Análisis de Resultados
El modelo detectó que los clientes entre 45 y 60 años presentan la tasa más alta de abandono.

Los miembros activos (IsActiveMember) tienen una probabilidad de fuga significativamente menor, lo que valida las estrategias de engagement.

Alemania se identificó como el mercado con mayor capital en riesgo según el análisis geográfico.

👨‍💻 Autor
Ignacio - Ingeniero en Informática (Inacap)
Especialista en Análisis de Datos y Desarrollo de Soluciones Técnicas.