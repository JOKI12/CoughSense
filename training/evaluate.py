"""
Script d'évaluation autonome des modèles entraînés pour CoughSense.
"""

import json
from pathlib import Path
import sys
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def evaluate_saved_model(
    model_name: str = "best",
    test_csv_path: Optional[Path] = None
) -> Dict:
    """Évalue un modèle sauvegardé sur un jeu de données de test."""
    if test_csv_path is None:
        test_csv_path = DATA_PROCESSED_DIR / "features_test.csv"

    if not test_csv_path.exists():
        raise FileNotFoundError(f"Fichier de test introuvable : {test_csv_path}")

    # Chargement du scaler et des noms de features
    scaler_path = MODELS_DIR / "scaler.joblib"
    feat_names_path = MODELS_DIR / "feature_names.json"

    if not (scaler_path.exists() and feat_names_path.exists()):
        raise FileNotFoundError("Artefacts d'entraînement manquants dans models/.")

    scaler = joblib.load(scaler_path)
    with open(feat_names_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # Chargement du modèle
    if model_name == "best":
        model_file = MODELS_DIR / "best_model.joblib"
    else:
        model_file = MODELS_DIR / f"{model_name.lower()}_model.joblib"

    if not model_file.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_file}")

    model = joblib.load(model_file)

    # Chargement des données de test
    df_test = pd.read_csv(test_csv_path)
    X_test_raw = df_test[feature_cols].values
    y_test = df_test["label"].values

    X_test = scaler.transform(X_test_raw)
    y_pred = model.predict(X_test)

    target_names = ["Healthy", "Respiratory_Abnormality"]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"==================================================================")
    print(f" Évaluation du modèle : {model_name}")
    print(f"==================================================================")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Matrice de confusion :")
    print(cm)

    return {
        "model_name": model_name,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "best"
    evaluate_saved_model(name)
