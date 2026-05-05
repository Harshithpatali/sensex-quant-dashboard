import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from preprocess import load_data, preprocess
from features import create_features
from pca_module import apply_pca

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "stock_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

df = load_data(DATA_PATH)
df = preprocess(df)

X, y = create_features(df)

X_pca = apply_pca(X, MODEL_DIR)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=100)
lr = LinearRegression()
xgb = XGBRegressor()

rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
xgb.fit(X_train, y_train)

joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(lr, os.path.join(MODEL_DIR, "lr_model.pkl"))
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

print("✅ Regression models trained!")