# run_preprocessor.py
from app.services.preprocessor import (
    transform_regression,
    transform_classification
)

# Payload de ejemplo:
payload = {
    "autoID": "7590-VHVEG",
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
    "Contract": "Two year",
    "PaymentMethod": "Electronic check",
    "Charges": 56.95,
}

print("=== Regresión – features finales ===")
print(transform_regression(payload))
print()

print("=== Clasificación – features finales ===")
print(transform_classification(payload))
