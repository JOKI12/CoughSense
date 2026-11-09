"""
Générateur de notebooks Jupyter d'exploration et de démonstration pour CoughSense.
"""

import json
from pathlib import Path


NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook(cells: list, output_path: Path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Notebook généré : {output_path}")


# Notebook 1 : Dataset Exploration
nb1_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# CoughSense — Notebook 01 : Exploration des Jeux de Données Réels\n",
            "**Master S3 Traitement du Signal**\n\n",
            "Ce notebook explore les métadonnées officielles et la distribution des participants issus des bases de données **Coswara (IISc Bangalore)** et **COUGHVID (EPFL)**."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import json\n",
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n\n",
            "# Chargement du résumé consolidé\n",
            "summary_path = Path('../data/processed/dataset_summary.json')\n",
            "if summary_path.exists():\n",
            "    with open(summary_path) as f:\n",
            "        summary = json.load(f)\n",
            "    print('Résumé du jeu de données :', json.dumps(summary, indent=2))\n",
            "else:\n",
            "    print('Veuillez d\\'abord exécuter : python training/prepare_dataset.py')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Analyse des splits\n",
            "train_df = pd.read_csv('../data/processed/features_train.csv')\n",
            "test_df = pd.read_csv('../data/processed/features_test.csv')\n",
            "print('Train shape:', train_df.shape)\n",
            "print('Test shape:', test_df.shape)\n\n",
            "# Vérification de l'étanchéité des participants (Anti-Data Leakage)\n",
            "train_parts = set(train_df['participant_id'])\n",
            "test_parts = set(test_df['participant_id'])\n",
            "overlap = train_parts & test_parts\n",
            "print('Nombre de participants partagés entre Train et Test :', len(overlap))\n",
            "assert len(overlap) == 0, 'Data leakage détecté !'"
        ]
    }
]

# Notebook 2 : Signal Processing
nb2_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# CoughSense — Notebook 02 : Traitement du Signal & Analyse Spectrale\n",
            "**Master S3 Traitement du Signal**\n\n",
            "Ce notebook détaille les étapes de filtrage Butterworth, débruitage par soustraction spectrale, FFT, STFT, Mel-Spectrogramme et extraction des 13 MFCCs."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.insert(0, '..')\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from src.audio_loader import load_and_standardize_audio\n",
            "from src.preprocessing import preprocess_signal\n",
            "from src.cough_detection import segment_cough\n",
            "from src.signal_analysis import analyze_signal\n",
            "from src.features import extract_features\n\n",
            "# Recherche d'un fichier audio réel\n",
            "sample_files = list(Path('../data/raw').glob('**/*cough*.wav'))\n",
            "audio_path = sample_files[0]\n",
            "print('Fichier audio sélectionné :', audio_path)\n\n",
            "audio_data = load_and_standardize_audio(audio_path, target_sample_rate=16000)\n",
            "prep = preprocess_signal(audio_data.signal, sample_rate=16000)\n",
            "seg = segment_cough(prep.processed_signal, sample_rate=16000)\n",
            "analysis = analyze_signal(seg.segmented_signal, sample_rate=16000)\n",
            "features = extract_features(seg.segmented_signal, sample_rate=16000)\n\n",
            "print('Durée analysée :', round(analysis.time_domain.duration_sec, 2), 's')\n",
            "print('Fréquence dominante :', round(analysis.frequency_domain.dominant_frequency_hz, 1), 'Hz')\n",
            "print('Centroïde spectral :', round(analysis.frequency_domain.spectral_centroid_hz, 1), 'Hz')\n",
            "print('Total des descripteurs extraits :', len(features.feature_names))"
        ]
    }
]

# Notebook 3 : Model Training
nb3_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# CoughSense — Notebook 03 : Entraînement & Comparaison des Modèles ML\n",
            "**Master S3 Traitement du Signal**\n\n",
            "Ce notebook charge les métriques réelles obtenues sur le test set pour comparer les performances de classification du SVM, Random Forest, KNN et CNN."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "import pandas as pd\n",
            "from pathlib import Path\n\n",
            "metrics_path = Path('../models/model_metrics.json')\n",
            "with open(metrics_path, 'r') as f:\n",
            "    metrics = json.load(f)\n\n",
            "summary_rows = []\n",
            "for model, m in metrics.items():\n",
            "    summary_rows.append({\n",
            "        'Modèle': model,\n",
            "        'Accuracy': f\"{m['accuracy']*100:.2f} %\",\n",
            "        'Précision': f\"{m.get('precision_macro', 0):.4f}\",\n",
            "        'Rappel': f\"{m.get('recall_macro', 0):.4f}\",\n",
            "        'F1-Score': f\"{m.get('f1_macro', 0):.4f}\",\n",
            "        'ROC-AUC': f\"{m.get('roc_auc', 0):.4f}\",\n",
            "        'Latence (ms)': f\"{m.get('inference_time_ms_per_sample', 0):.3f} ms\"\n",
            "    })\n\n",
            "df_compare = pd.DataFrame(summary_rows)\n",
            "print(df_compare.to_string(index=False))"
        ]
    }
]

create_notebook(nb1_cells, NOTEBOOKS_DIR / "01_dataset_exploration.ipynb")
create_notebook(nb2_cells, NOTEBOOKS_DIR / "02_signal_processing.ipynb")
create_notebook(nb3_cells, NOTEBOOKS_DIR / "03_model_training.ipynb")
