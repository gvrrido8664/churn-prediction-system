# 🛡️ Sistema de Predicción de Fuga de Clientes (Churn)
> **Proyecto de Inteligencia Artificial y Analítica Predictiva**

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Power_BI](https://img.shields.io/badge/Power_BI-Analytics-yellow?style=for-the-badge&logo=powerbi&logoColor=black)

## 📋 Descripción del Proyecto
Este sistema soluciona uno de los problemas financieros más críticos: la **fuga de clientes**. Utilizando un modelo de **Machine Learning (Random Forest)**, el software analiza el comportamiento histórico para predecir qué clientes tienen mayor probabilidad de abandonar la institución.

Los resultados son procesados automáticamente y visualizados en un **Dashboard de Alerta Temprana**, permitiendo que el equipo comercial tome acciones preventivas antes de que el cliente se retire.

---

## 🚀 Características Principales
* **Modelo Predictivo:** Clasificador entrenado con un **86.60% de exactitud**.
* **Análisis Geográfico:** Identificación de mercados críticos mediante Treemaps dinámicos.
* **Priorización:** Tabla de acción inmediata con "Score de Riesgo" y formato condicional (Semáforo).
* **Pipeline Automatizado:** Script en Python con rutas dinámicas para procesar nuevos datos.

---

## 📁 Estructura del Repositorio
Para facilitar la navegación del equipo técnico, el proyecto se organiza así:

| Carpeta | Contenido |
| :--- | :--- |
| `📂 data` | Datasets originales y procesados (`clientes_con_riesgo.csv`). |
| `📂 notebooks` | Análisis Exploratorio (EDA) y entrenamiento del modelo. |
| `📂 models` | Modelos serializados (`.pkl`) y metadatos del entrenamiento. |
| `📂 scripts` | Lógica de inferencia y procesamiento automatizado. |
| `📂 dashboard` | Archivo `.pbix` con la visualización estratégica final. |

---

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python (Pandas, Numpy).
* **IA:** Scikit-Learn (Random Forest Classifier).
* **Visualización:** Power BI & DAX (Medidas de riesgo y salud de cartera).
* **Entorno:** VS Code & Jupyter Notebooks.

---

## ⚙️ Instrucciones de Uso

1. **Instalar dependencias:**
   ```bash
   pip install pandas scikit-learn joblib
   ```

2. **Generar predicciones:**
   Ejecuta el script desde la raíz para obtener el archivo de riesgo:
   ```bash
   python scripts/predict_churn.py
   ```

3. **Explorar el Dashboard:**
   Abre el archivo en `dashboard/` y conecta la fuente de datos al nuevo CSV generado en la carpeta `data/`.

---

## 📈 Conclusiones Técnicas
* **Variable Crítica:** La **edad** es el predictor con mayor peso, concentrando el riesgo en el segmento de 45-60 años.
* **Engagement:** El estatus de **Miembro Activo** reduce drásticamente la probabilidad de fuga, sugiriendo que las estrategias de fidelización deben enfocarse en el uso constante de productos.

---

### 👤 Autor
**Ignacio** - *Ingeniero en Informática (Inacap)*
