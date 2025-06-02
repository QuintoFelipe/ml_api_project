# run_predictor_test.py

"""
Script de prueba para verificar que el método predict() mapea correctamente
la salida de clasificación 0/1 a “Alpha”/“Betha”, sin necesidad de cargar modelos reales.
"""

from app.services.predictor import predict
import app.services.loader as loader_module

# 1) Definimos payload de ejemplo (puedes cambiar valores a tu gusto):
payload = {
    "autoID": "TEST-1234",
    "SeniorCity": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "Service1": "Yes",
    "Service2": "No phone service",
    "Security": "No internet service",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No internet service",
    "TechSupport": "No",
    "PaperlessBilling": "Yes",
    "Contract": "One year",
    "PaymentMethod": "Credit card (automatic)",
    "Charges": 45.00,
}

# 2) Creamos “modelos dummy” que simulan la interfaz de un clasificador y un regresor.
#    - El regresor siempre devolverá un valor fijo (ej. 123.45).
#    - El clasificador devolverá 0 o 1, según lo forcemos.

class DummyRegressor:
    def predict(self, df):
        # siempre regresará [123.45]
        return [123.45]

class DummyClassifierAlpha:
    def predict(self, df):
        return [0]            # fuerza clase 0 → mapeo "Alpha"
    def predict_proba(self, df):
        # Ejemplo de salida: [prob_clase_0, prob_clase_1]
        return [[0.7, 0.3]]   # prob de “Betha” (1) es 0.3

class DummyClassifierBetha:
    def predict(self, df):
        return [1]            # fuerza clase 1 → mapeo "Betha"
    def predict_proba(self, df):
        return [[0.2, 0.8]]   # prob de “Betha” (1) es 0.8

# 3) Reemplazamos las funciones que “cargan” los modelos para que devuelvan los dummies.
#    De esta manera, `predict()` no cargará los .pkl reales, sino nuestros dummy.

# Primero: forzamos clasificación = 0 (Alpha)
loader_module.get_regression_model = lambda: DummyRegressor()
loader_module.get_classification_model = lambda: DummyClassifierAlpha()

print("### PRUEBA 1: clasificador devuelve 0 → debería mapear a 'Alpha' ###")
resultado_alpha = predict(payload)
print(resultado_alpha)
print()

# Segundo: forzamos clasificación = 1 (Betha)
loader_module.get_regression_model = lambda: DummyRegressor()
loader_module.get_classification_model = lambda: DummyClassifierBetha()

print("### PRUEBA 2: clasificador devuelve 1 → debería mapear a 'Betha' ###")
resultado_betha = predict(payload)
print(resultado_betha)
