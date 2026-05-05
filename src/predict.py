import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_models():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))

    rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    lr = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))

    return scaler, pca, rf, lr, xgb

def predict(input_data):
    scaler, pca, rf, lr, xgb = load_models()

    arr = np.array(input_data).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    arr_pca = pca.transform(arr_scaled)

    return {
        "RandomForest (%)": float(rf.predict(arr_pca)[0]),
        "LinearRegression (%)": float(lr.predict(arr_pca)[0]),
        "XGBoost (%)": float(xgb.predict(arr_pca)[0])
    }