"""
Module et script de comparaison des modèles entraînés pour CoughSense.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def get_model_comparison_dataframe() -> pd.DataFrame:
    """Charge les métriques réelles de tous les modèles et retourne un tableau comparatif."""
    metrics_path = MODELS_DIR / "model_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Métriques de modèles introuvables. "
            "Exécutez d'abord : python training/train.py"
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    rows = []
    for model_name, m in metrics.items():
        rows.append({
            "Modèle": model_name,
            "Exactitude (Accuracy)": f"{m.get('accuracy', 0.0) * 100:.2f} %",
            "Exactitude Équilibrée": f"{m.get('balanced_accuracy', 0.0) * 100:.2f} %",
            "Précision Macro": f"{m.get('precision_macro', 0.0):.4f}",
            "Rappel Macro": f"{m.get('recall_macro', 0.0):.4f}",
            "Score F1 Macro": f"{m.get('f1_macro', 0.0):.4f}",
            "ROC-AUC": f"{m.get('roc_auc', 0.0):.4f}",
            "Temps Inférence (ms)": f"{m.get('inference_time_ms_per_sample', 0.0):.3f} ms"
        })

    return pd.DataFrame(rows)


def main():
    print("==================================================================")
    print(" CoughSense - Comparaison des Modèles sur Données Réelles de Test")
    print("==================================================================")
    try:
        df = get_model_comparison_dataframe()
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Erreur : {e}")


if __name__ == "__main__":
    main()
