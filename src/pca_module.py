from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

def apply_pca(X, model_dir, n_components=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(pca, os.path.join(model_dir, "pca.pkl"))

    return X_pca