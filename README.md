# ML API Project

Este repositorio reúne dos componentes principales que apoyan el análisis y la predicción de demanda para Cementos Argos:

1. **API (FastAPI)** en `app/` que expone modelos de clasificación y regresión en tiempo real para una **aplicación de experiencia del colaborador**.  
2. **Laboratorios (LAB_ML)** en `LAB_ML/` que contienen notebooks, datos y resultados de los procesos de entrenamiento y forecasting.

---

## Introducción a la API

La API sirve como **backend** para una aplicación frontend de experiencia del colaborador. En dicha aplicación, el usuario selecciona parámetros (por ejemplo, tipo de servicio, flags “Yes/No”, método de pago, etc.) desde listas desplegables, y envía estos datos en formato JSON. El propósito de este servicio es:

- **Preprocesar** automáticamente esos datos (codificaciones, flags, ingeniería de features).  
- **Clasificar** el caso en uno de dos códigos (`Alpha` o `Betha`), que guían al analista sobre qué tipo de compras de materiales debe realizar y en qué cantidades.  
- **Registrar la probabilidad** asociada a la etiqueta predicha.  
- **Predecir la demanda numérica** de compras de Cementos Argos para el conjunto de características ingresadas.  

Al final, la API devuelve un JSON con:

```
{
  "autoID": "id_unico",
  "class_pred": "Alpha" | "Betha",
  "class_prob": 0.0 - 1.0,
  "demand_pred": 0.0 - …
}
```
De esta forma, el frontend recibe en tiempo real tanto el **código de clasificación (Alpha/Betha)** como el **valor estimado de demanda**, permitiendo decisiones de compra informadas.

## 1. Estructura general del repositorio
```
C:.
│   Dockerfile
│   README.md
│   requirements.txt
│   teoria_.pdf
│   to_predict.csv
│
├───app/                       ← Código fuente de la API
│   │   main.py
│   │   __init__.py
│   │
│   ├───api/
│   │   ├── deps.py
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── endpoints.py   ← POST /v1/predict
│   │       └── __init__.py
│   │
│   ├───core/
│   │   ├── config.py          ← Variables de entorno (Pydantic)
│   │   ├── logging.py         ← Configuración de logs
│   │   ├── version.py         ← Versión de la API
│   │   └── __init__.py
│   │
│   ├───models/
│   │   ├── request.py         ← Esquema Pydantic de entrada
│   │   └── response.py        ← Esquema Pydantic de salida
│   │
│   └───services/
│       ├── loader.py          ← Carga de pipelines (.pkl)
│       ├── predictor.py       ← Lógica de predicción (clasificación + regresión)
│       ├── preprocessor.py    ← Transformaciones del JSON a DataFrame
│       └── __init__.py
│
├───models/                    ← Pipelines serializados que usa la API
│       catboost_reg_pipeline.pkl     # Pipeline completo de regresión
│       logreg_clf_pipeline.pkl       # Pipeline completo de clasificación
│
├───LAB_ML/                    ← Laboratorio de entrenamiento y forecasting
│   ├───Clasificacion_Regresion/
│   │   ├───models/
│   │   │       catboost_reg_pipeline.joblib
│   │   │       logistic_clf_pipeline.joblib
│   │   │       txt_info_modelos.json
│   │   │
│   │   ├───notebooks/
│   │   │       EDA_&_preprosesado.ipynb
│   │   │       ML01_experiment_class_regresion.ipynb
│   │   │
│   │   ├───raw_data/
│   │   │       dataset_alpha_betha.csv
│   │   │
│   │   └───transformed_data/
│   │           dataset_A_raw.csv
│   │           dataset_B_addon_sum.csv
│   │           dataset_C_contractx.csv
│   │           dataset_D_fulleng.csv
│   │
│   └───Prediccion_de_demanda/
│       ├───notebook/
│       │       demand_forecasting_notebook.ipynb
│       │
│       ├───raw_data/
│       │       dataset_demand_acumulate.csv
│       │
│       └───resultados/
│               cementos_argos_demand_forecast_2017_2022.csv
│               demanda_argos_train_valid_forecast.png

```

- **root/**
    - `Dockerfile` : instrucciones para empacar la API en Docker.
    - `requirements.txt` : dependencias Python necesarias para la API.
    - `teoria_.pdf` : respuestas al examen teorico.
    - `to_predict.csv `: archivo CSV con los registros para los cuales se solicitaba la prediccion
- **app/**
    - Contiene toda la lógica de la API FastAPI.
- **models/**
    - Pipelines serializados (`.pkl`) que utiliza la API para inferencia.
- **LAB_ML/**
    - **Clasificacion_Regresion/:** notebooks, datos y artefactos que ilustran cómo se desarrollaron los pipelines de regresión y clasificación.
    - **Prediccion_de_demanda/:** notebooks, datos y resultados de un proyecto de forecasting de la demanda histórica (serie temporal).

## 2. API (FastAPI)
### **2.1. Descripción general**
- **Endpoint principal:** ``POST /v1/predict.``

- **Entrada:** JSON con campos validados por Pydantic (ver ``app/models/request.py``).

- **Proceso:**

    1. ``preprocessor.py`` convierte el JSON en un ``DataFrame`` listo para los pipelines:

        - Crea columnas binarias, reemplaza textos (“No internet service” → “No”, “No phone service” → “No”), genera ``Contract_months``, ``InternetService``, ``AutoPayment_flag`` y ``PaymentMethod_simple``.

        - Para clasificación, agrega ``TotalAddOns``, ``Charges_per_AddOn`` y ``Contract_x_Charges``.

    2. ``predictor.py`` aplica:

        - **Clasificación** (``logreg_clf_pipeline.pkl``): devuelve ``0`` o ``1`` → se mapea a ``"Alpha"``/``"Betha"``. Al mismo tiempo, extrae la probabilidad de la clase predicha.

        - **Regresión** (``catboost_reg_pipeline.pkl``): devuelve un valor numérico de demanda.
    3. Se retorna un JSON como:

```json
    {
  "autoID": "7590-VHVEG",
  "class_pred": "Alpha",
  "class_prob": 0.81,
  "demand_pred": 55.2
}
```
### 2.2 Dependencias
Listadas en ``requirements.txt`` 

## 3. Despliegue de la API con Docker
### 3.1. Construir la imagen
Desde la raíz (``C:\…\ML_API_PROJECT``), ejecuta:

```bash
docker build -t ml_api_ml:latest .
```
- Utiliza ``python:3.11-slim`` como base.
- Instala paquetes de ``requirements.txt.``
- Copia la carpeta ``app/`` y los pipelines de ``models/`` en la imagen.

### 3.2. Ejecutar el contenedor
```bash
docker run -d \
  --name ml_api_container \
  -p 8000:8000 \
  ml_api_ml:latest
```
- ``-d`` corre el contenedor en segundo plano.
- ``--name ml_api_container``: nombre identificador del contenedor.
- ``-p 8000:8000``: mapea puerto 8000 del contenedor a 8000 del host.

### 3.3. Verificar y probar
 **1.** Comprueba el contenedor en ejecución:
```bash
docker ps
```
**2.** Accede a Swagger UI:
```bash
http://127.0.0.1:8000/docs
```
**3.** Ejemplo con ``curl``:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
        "autoID": "7590-VHVEG",
        "SeniorCity": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Service1": "Yes",
        "Service2": "No phone service",
        "Security": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "PaperlessBilling": "Yes",
        "Contract": "Two year",
        "PaymentMethod": "Electronic check",
        "Charges": 56.95
      }'
```
Respuesta esperada:
```json
{
  "autoID": "7590-VHVEG",
  "class_pred": "Alpha",
  "class_prob": 0.81,
  "demand_pred": 55.2
}
```
## 4. Ejecución local (sin Docker)

Si no deseas usar Docker, sigue estos pasos:

**1.** Crea y activa un entorno Conda con Python 3.11:
```bash
conda create -n ml_api python=3.11 -y
conda activate ml_api
```

**2.** Instala dependencias:
```bash
pip install -r requirements.txt
pip install scikit-learn==1.6.1 pydantic-settings
```
**3.** Levanta el servidor:
```bash
uvicorn app.main:app --reload
```
**4.** Abre ``http://127.0.0.1:8000/docs`` para probar el endpoint ``POST /v1/predict``.

## 5. LAB_ML: Entrenamiento y Forecasting
Dentro de ``LAB_ML/`` están los notebooks y datos de desarrollo original:
- **Clasificacion_Regresion/**
    - ``models/``: pipelines ``.joblib`` y metadatos en ``txt_info_modelos.json``.
    - ``notebooks/:``
        - ``EDA_&_preprosesado.ipynb``: limpieza, EDA y generación de CSVs transformados.
        - ``ML01_experiment_class_regresion.ipynb``: comparación de bloques A–D, ajuste final de CatBoost y LogisticRegression, extracción de métricas.

- ``raw_data/dataset_alpha_betha.csv``: datos originales de clasificación.
- ``transformed_data/``: CSVs con cada bloque de ingeniería de features (A, B, C, D).

- **Prediccion_de_demanda/**

    - ``notebook/demand_forecasting_notebook.ipynb:`` flujo completo de forecast (SARIMA).
    - ``raw_data/dataset_demand_acumulate.csv:`` serie histórica de demanda.
    - ``resultados/``:
        - ``cementos_argos_demand_forecast_2017_2022.csv:`` fechas, valores reales y pronósticos (ene-abr 2022, may-jul 2022).

        - ``demanda_argos_train_valid_forecast.png``: gráfico final con  entrenamiento, validación y pronóstico.

> **Nota:** Todo en LAB_ML/ es para referencia y trazabilidad de cómo se crearon los modelos que ahora utiliza la API. La API consume únicamente los archivos serializados en models/. El modelo de ``Prediccio_de_demanda`` no se usa dentro de la API

## 6. Archivos adicionales en la raíz
- ``teoria_.pdf``: Documento teórico de apoyo (explica fundamentos generales).

- ``to_predict.csv``: CSV con registros para los cuales se desean ejecutar predicciones usando la API (puede cargarse y procesarse mediante scripts o directamente enviar al endpoint).

## 7. Autor y Contacto
**Juan Felipe Quinto Ríos**
| Científico de Datos |
 quintoriosjuanfelip@gmail.com

 
