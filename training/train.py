"""
Pipeline d'entraînement et d'évaluation des modèles de Machine Learning pour CoughSense.

Modèles implémentés :
1. Support Vector Machine (SVM) à noyau RBF avec calibration probabiliste
2. Forêt Aléatoire (Random Forest) avec pondération de classe équilibrée
3. K-Plus Proches Voisins (K-Nearest Neighbors - KNN) avec métrique de distance

Métriques évaluées sur le jeu de test indépendant (held-out test set) :
- Exactitude (Accuracy)
- Exactitude équilibrée (Balanced Accuracy)
- Précision macro et par classe
- Rappel / Sensibilité macro et par classe
- Score F1 macro et par classe
- Matrice de confusion
- Aire sous la courbe ROC (ROC-AUC)
- Temps d'inférence moyen par échantillon (ms)
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


METADATA_COLS = [
    "file_path",
    "participant_id",
    "dataset",
    "raw_status",
    "label",
    "label_name",
    "split",
    "cough_detected",
    "num_bursts",
    "snr_db",
    "original_duration_sec",
    "processed_duration_sec"
]


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Charge les sous-ensembles d'apprentissage et extrait la liste des caractéristiques."""
    train_path = DATA_PROCESSED_DIR / "features_train.csv"
    val_path = DATA_PROCESSED_DIR / "features_val.csv"
    test_path = DATA_PROCESSED_DIR / "features_test.csv"

    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            "Fichiers d'entraînement introuvables. "
            "Exécutez d'abord : python training/prepare_dataset.py"
        )

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path) if val_path.exists() else pd.DataFrame()
    df_test = pd.read_csv(test_path)

    # Identification des colonnes de caractéristiques
    feature_cols = [c for c in df_train.columns if c not in METADATA_COLS]

    return df_train, df_val, df_test, feature_cols


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str]
) -> Dict:
    """Évalue un modèle sur le jeu de test indépendant et mesure la latence d'inférence."""
    # Mesure du temps d'inférence
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    t1 = time.perf_counter()
    latency_ms = ((t1 - t0) / len(X_test)) * 1000.0

    # Probabilités pour ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "roc_auc": round(auc, 4),
        "inference_time_ms_per_sample": round(latency_ms, 3),
        "confusion_matrix": cm,
        "classification_report": clf_report
    }


def train_and_evaluate_all():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" CoughSense - Entraînement des Modèles & Évaluation Rigoureuse")
    print("==================================================================")

    # 1. Chargement des données
    df_train, df_val, df_test, feature_cols = load_datasets()

    # Si un ensemble de validation existe, on peut le combiner pour le ré-entraînement final
    # ou entraîner sur train et valider sur test
    X_train_raw = df_train[feature_cols].values
    y_train = df_train["label"].values

    X_test_raw = df_test[feature_cols].values
    y_test = df_test["label"].values

    target_names = ["Healthy", "Respiratory_Abnormality"]

    print(f"Caractéristiques d'entrée : {len(feature_cols)}")
    print(f"Échantillons d'entraînement : {len(X_train_raw)} (Sain: {sum(y_train==0)}, Anomalie: {sum(y_train==1)})")
    print(f"Échantillons de test        : {len(X_test_raw)} (Sain: {sum(y_test==0)}, Anomalie: {sum(y_test==1)})")

    # 2. Normalisation standard (Ajustée STRICTEMENT sur l'ensemble d'entraînement)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # 3. Définition des modèles
    models = {
        "SVM": SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski"
        )
    }

    metrics_results = {}
    best_model_name = None
    best_f1 = -1.0

    print("\n------------------------------------------------------------------")
    print(" Entraînement & Évaluation sur Held-Out Test Set")
    print("------------------------------------------------------------------")

    for name, clf in models.items():
        print(f"\n--- Modèle : {name} ---")
        clf.fit(X_train, y_train)

        # Évaluation sur le jeu de test
        eval_metrics = evaluate_model(clf, X_test, y_test, target_names)
        metrics_results[name] = eval_metrics

        # Sauvegarde du modèle individuel
        model_file = MODELS_DIR / f"{name.lower()}_model.joblib"
        joblib.dump(clf, model_file)

        print(f"Accuracy         : {eval_metrics['accuracy'] * 100:.2f} %")
        print(f"Balanced Acc.    : {eval_metrics['balanced_accuracy'] * 100:.2f} %")
        print(f"Precision Macro  : {eval_metrics['precision_macro']:.4f}")
        print(f"Recall Macro     : {eval_metrics['recall_macro']:.4f}")
        print(f"F1-Score Macro   : {eval_metrics['f1_macro']:.4f}")
        print(f"ROC-AUC          : {eval_metrics['roc_auc']:.4f}")
        print(f"Inference Latency: {eval_metrics['inference_time_ms_per_sample']:.3f} ms/échantillon")
        print("Matrice de Confusion :")
        print(f"  [TN={eval_metrics['confusion_matrix'][0][0]}, FP={eval_metrics['confusion_matrix'][0][1]}]")
        print(f"  [FN={eval_metrics['confusion_matrix'][1][0]}, TP={eval_metrics['confusion_matrix'][1][1]}]")

        if eval_metrics["f1_macro"] > best_f1:
            best_f1 = eval_metrics["f1_macro"]
            best_model_name = name

    # 4. Sauvegarde du meilleur modèle
    best_model = models[best_model_name]
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    best_info = {
        "best_model_name": best_model_name,
        "best_f1_score": best_f1,
        "selected_criterion": "f1_macro",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(MODELS_DIR / "best_model_info.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)

    # 5. Sauvegarde des métriques globales et noms de caractéristiques
    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_results, f, indent=2)

    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    print("\n==================================================================")
    print(f" MEILLEUR MODÈLE RETENU : {best_model_name} (F1 = {best_f1:.4f})")
    print(f" Modèles et métriques enregistrés dans : {MODELS_DIR.resolve()}")
    print("==================================================================")


if __name__ == "__main__":
    train_and_evaluate_all()
