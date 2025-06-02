import pandas as pd
from app.services.preprocessor import transform

def test_transform_shape_and_columns():
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
    df = transform(payload)
    assert list(df.columns) == [
        "SeniorCity", "Partner", "Dependents", "Service1", "Service2",
        "Security", "OnlineBackup", "DeviceProtection", "TechSupport",
        "PaperlessBilling", "InternetService", "Charges", "Contract_months",
        "AutoPayment_flag", "PaymentMethod_simple",
    ]
    # Valores clave
    assert df.loc[0, "InternetService"] == 0
    assert df.loc[0, "Service2"] == 0
    assert df.loc[0, "Security"] == 0
    assert df.loc[0, "Contract_months"] == 24
    assert df.loc[0, "AutoPayment_flag"] == 0