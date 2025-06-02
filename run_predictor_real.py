# run_predictor_real.py

"""
Script de prueba que usa los modelos serializados en /models para:
  1) cargar el modelo de clasificación y regresión reales
  2) preprocesar un payload de ejemplo
  3) imprimir la predicción real (class_pred, class_prob, demand_pred)
"""

import warnings
from pprint import pprint

# Suprime las advertencias de “InconsistentVersionWarning” cuando carga los .pkl
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from app.services import predictor

def main():
    # Payload de ejemplo (ajústalo si cambian tus features)
    payload = {
        "autoID": "REAL-TEST-0001",
        "SeniorCity": 1,
        "Partner": "Yes",
        "Dependents": "No",
        "Service1": "No",
        "Service2": "Yes",
        "Security": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "PaperlessBilling": "Yes",
        "Contract": "Month-to-month",
        "PaymentMethod": "Bank transfer (automatic)",
        "Charges": 99.99,
    }

    print("Payload de prueba:")
    pprint(payload)
    print("\n--- Ejecutando predict() con modelos reales…\n")

    try:
        resultado = predictor.predict(payload)
    except Exception as e:
        print("Error al invocar predictor.predict():\n", e)
        return

    print("Resultado de la predicción real:")
    pprint(resultado)

if __name__ == "__main__":
    main()
