# run_predictor_test.py

"""
Script de prueba para verificar que el método predict() mapea correctamente
la salida de clasificación 0/1 a “Alpha”/“Betha”, SIN cargar los .pkl reales.
"""

# 1) Importa predictor _antes_ de hacer monkeypatch, 
#    para que sus referencias a get_<…>Model se puedan sobreescribir.
from app.services import predictor

# 2) Definimos payload de ejemplo (idéntico al que usa tu API)
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

# 3) Clases dummy que reemplazarán los modelos reales:
class DummyRegressor:
    def predict(self, df):
        return [123.45]  # valor fijo de demanda

class DummyClassifierAlpha:
    def predict(self, df):
        return [0]            # fuerza clase 0 (Alpha)
    def predict_proba(self, df):
        # Simulamos probabilidad [p(0), p(1)]
        return [[0.7, 0.3]]   # prob de “Betha” (1) es 0.3

class DummyClassifierBetha:
    def predict(self, df):
        return [1]            # fuerza clase 1 (Betha)
    def predict_proba(self, df):
        return [[0.2, 0.8]]   # prob de “Betha” (1) es 0.8

# 4) Monkeypatch: sustituimos las funciones que cargan los modelos en predictor
predictor.get_regression_model = lambda: DummyRegressor()
predictor.get_classification_model = lambda: DummyClassifierAlpha()

print("=== PRUEBA 1: Clasificador devuelve 0 → debería mapear a 'Alpha' ===")
resultado_alpha = predictor.predict(payload)
print(resultado_alpha)
print()

# 5) Ahora forzamos la salida 1 → “Betha”
predictor.get_classification_model = lambda: DummyClassifierBetha()

print("=== PRUEBA 2: Clasificador devuelve 1 → debería mapear a 'Betha' ===")
resultado_betha = predictor.predict(payload)
print(resultado_betha)
