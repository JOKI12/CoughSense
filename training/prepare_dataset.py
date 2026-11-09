"""
Script de préparation et d'extraction des caractéristiques pour CoughSense.

Responsabilités :
1. Analyse et validation des métadonnées officielles (Coswara & COUGHVID).
2. Définition de la tâche de classification (Sain / Normal vs Anomalie respiratoire).
3. Découpage stratifié STRICT par participant (Prévention absolue du Data Leakage).
4. Traitement du signal standardisé :
   - Décodage et rééchantillonnage à 16 kHz
   - Débruitage par soustraction spectrale
   - Filtrage passe-bande Butterworth [80 Hz, 4000 Hz]
   - Normalisation d'amplitude
   - Segmentation de la toux
   - Extraction des 91 descripteurs acoustiques & cepstraux (MFCC)
5. Enregistrement des fichiers d'entraînement, validation et test dans data/processed/.
"""

import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tqdm import tqdm

# Ajout de la racine du projet au sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_loader import AudioValidationError, load_and_standardize_audio
from src.cough_detection import segment_cough
from src.features import extract_features
from src.preprocessing import preprocess_signal


DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def map_clinical_status(raw_status: str) -> Optional[Tuple[int, str]]:
    """
    Associe le statut clinique officiel Coswara à une classe binaire d'expérimentation.

    Classe 0 : 'Healthy' (Sujet sain / normal)
    Classe 1 : 'Respiratory_Abnormality' (Infection ou pathologie respiratoire active)
    """
    raw_status = str(raw_status).strip().lower()

    if raw_status in ["healthy", "no_resp_illness_exposed"]:
        return 0, "Healthy"
    elif raw_status in [
        "positive_mild",
        "positive_moderate",
        "resp_illness_not_identified",
        "positive_asymp"
    ]:
        return 1, "Respiratory_Abnormality"
    else:
        # Exclure les cas ambigus (ex: recovered_full ou under_validation) pour pureté de dépistage
        return None


def collect_raw_samples() -> pd.DataFrame:
    """
    Parcourt le répertoire data/raw/ pour identifier les enregistrements de toux
    et extraire leurs métadonnées associées.
    """
    samples = []

    # 1. Parcours des enregistrements Coswara
    coswara_dir = DATA_RAW_DIR / "coswara"
    if coswara_dir.exists():
        for batch_dir in coswara_dir.iterdir():
            if not batch_dir.is_dir() or batch_dir.name in ["__pycache__", ".ipynb_checkpoints"]:
                continue

            for part_dir in batch_dir.iterdir():
                if not part_dir.is_dir():
                    continue

                part_id = part_dir.name
                meta_json = part_dir / "metadata.json"

                raw_status = None
                age = np.nan
                gender = "unknown"

                if meta_json.exists():
                    try:
                        with open(meta_json, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            raw_status = meta.get("covid_status")
                            age = meta.get("a", np.nan)
                            gender = meta.get("g", "unknown")
                    except Exception:
                        pass

                mapping = map_clinical_status(raw_status) if raw_status else None
                if mapping is None:
                    continue

                label_id, label_name = mapping

                for cough_file in part_dir.glob("cough-*.wav"):
                    samples.append({
                        "file_path": str(cough_file),
                        "participant_id": f"coswara_{part_id}",
                        "dataset": "Coswara",
                        "raw_status": raw_status,
                        "label": label_id,
                        "label_name": label_name,
                        "age": age,
                        "gender": gender,
                        "cough_type": cough_file.stem
                    })

    # 2. Parcours des enregistrements COUGHVID s'ils existent
    coughvid_dir = DATA_RAW_DIR / "coughvid"
    if coughvid_dir.exists():
        cough_wav = coughvid_dir / "cough.wav"
        if cough_wav.exists():
            samples.append({
                "file_path": str(cough_wav),
                "participant_id": "coughvid_official_sample_01",
                "dataset": "COUGHVID",
                "raw_status": "symptomatic_cough",
                "label": 1,
                "label_name": "Respiratory_Abnormality",
                "age": np.nan,
                "gender": "unknown",
                "cough_type": "cough"
            })

    return pd.DataFrame(samples)


def process_and_extract_features(df_samples: pd.DataFrame) -> pd.DataFrame:
    """
    Applique la chaîne de traitement du signal et extrait le vecteur de caractéristiques
    pour chaque fichier audio valide.
    """
    processed_rows = []

    print(f"\nTraitement du signal et extraction des caractéristiques ({len(df_samples)} fichiers)...")

    for idx, row in tqdm(df_samples.iterrows(), total=len(df_samples), desc="DSP & Features"):
        fpath = Path(row["file_path"])
        if not fpath.exists():
            continue

        try:
            # 1. Chargement et standardisation (16 kHz, mono, validation)
            audio_data = load_and_standardize_audio(
                fpath,
                target_sample_rate=16000,
                min_duration_sec=0.20,
                max_duration_sec=15.0
            )

            # 2. Prétraitement (débruitage spectral + filtre passe-bande 80-4000 Hz + normalisation)
            prep_res = preprocess_signal(
                audio_data.signal,
                sample_rate=16000,
                apply_noise_reduction=True,
                apply_filtering=True,
                low_cutoff_hz=80.0,
                high_cutoff_hz=4000.0,
                filter_order=4,
                target_peak=0.95
            )

            # 3. Segmentation du burst de toux
            seg_res = segment_cough(
                prep_res.processed_signal,
                sample_rate=16000,
                threshold_multiplier=1.20
            )

            # Signal final utilisé pour l'extraction de caractéristiques
            eval_signal = seg_res.segmented_signal if len(seg_res.segmented_signal) >= 3200 else prep_res.processed_signal

            # 4. Extraction complète des 91 descripteurs (temporels, spectraux, MFCCs)
            feat_res = extract_features(
                eval_signal,
                sample_rate=16000,
                n_mfcc=13,
                include_deltas=True
            )

            # Agrégation des métadonnées et caractéristiques
            record = {
                "file_path": str(fpath),
                "participant_id": row["participant_id"],
                "dataset": row["dataset"],
                "raw_status": row["raw_status"],
                "label": int(row["label"]),
                "label_name": row["label_name"],
                "cough_detected": int(seg_res.cough_detected),
                "num_bursts": int(seg_res.num_bursts),
                "snr_db": float(prep_res.snr_estimated_db),
                "original_duration_sec": float(audio_data.original_duration),
                "processed_duration_sec": float(len(eval_signal) / 16000.0)
            }

            # Fusionner les caractéristiques extraites
            record.update(feat_res.feature_dict)
            processed_rows.append(record)

        except (AudioValidationError, Exception) as err:
            # Rejeter silencieusement ou avertir pour les fichiers corrompus/inaudibles
            continue

    return pd.DataFrame(processed_rows)


def split_by_participant(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Effectue un découpage Train / Validation / Test stratifié par participant
    pour garantir l'absence totale de fuite de données (Data Leakage).
    """
    # Liste unique des participants avec leur classe majoritaire
    part_summary = df.groupby("participant_id")["label"].agg(lambda s: s.mode()[0]).reset_index()

    # 1. Séparation Train+Val vs Test par participant
    train_val_parts, test_parts = train_test_split(
        part_summary,
        test_size=test_size,
        stratify=part_summary["label"],
        random_state=random_seed
    )

    # 2. Séparation Train vs Val par participant
    relative_val_size = val_size / (1.0 - test_size)
    train_parts, val_parts = train_test_split(
        train_val_parts,
        test_size=relative_val_size,
        stratify=train_val_parts["label"],
        random_state=random_seed
    )

    train_set = set(train_parts["participant_id"])
    val_set = set(val_parts["participant_id"])
    test_set = set(test_parts["participant_id"])

    # Vérification stricte d'étanchéité
    assert len(train_set & test_set) == 0, "ERREUR CRITIQUE : Fuite de données entre Train et Test !"
    assert len(train_set & val_set) == 0, "ERREUR CRITIQUE : Fuite de données entre Train et Validation !"
    assert len(val_set & test_set) == 0, "ERREUR CRITIQUE : Fuite de données entre Validation et Test !"

    df_train = df[df["participant_id"].isin(train_set)].copy()
    df_val = df[df["participant_id"].isin(val_set)].copy()
    df_test = df[df["participant_id"].isin(test_set)].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    return df_train, df_val, df_test


def main():
    print("==================================================================")
    print(" CoughSense - Préparation du Jeu de Données & Extraction DSP")
    print("==================================================================")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collecte des échantillons
    df_raw = collect_raw_samples()
    if df_raw.empty:
        print("Erreur : Aucun fichier audio trouvé dans data/raw/.")
        print("Veuillez d'abord exécuter : python scripts/download_dataset.py")
        sys.exit(1)

    print(f"\nFichiers bruts admissibles identifiés : {len(df_raw)}")
    print(f"Nombre de participants uniques : {df_raw['participant_id'].nunique()}")
    print("\nRépartition par statut clinique :")
    print(df_raw["label_name"].value_counts())

    # 2. Découpage par participant (Anti Data-Leakage)
    df_features = process_and_extract_features(df_raw)

    if df_features.empty:
        print("Erreur : Aucun fichier n'a pu être prétraité.")
        sys.exit(1)

    df_train, df_val, df_test = split_by_participant(
        df_features,
        test_size=0.20,
        val_size=0.15,
        random_seed=42
    )

    # 3. Sauvegarde des datasets
    train_path = DATA_PROCESSED_DIR / "features_train.csv"
    val_path = DATA_PROCESSED_DIR / "features_val.csv"
    test_path = DATA_PROCESSED_DIR / "features_test.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    # Sauvegarde des métadonnées globales
    summary = {
        "total_recordings": int(len(df_features)),
        "total_participants": int(df_features["participant_id"].nunique()),
        "classes": {
            0: "Healthy",
            1: "Respiratory_Abnormality"
        },
        "class_distribution": {
            "Healthy": int(np.sum(df_features["label"] == 0)),
            "Respiratory_Abnormality": int(np.sum(df_features["label"] == 1))
        },
        "splits": {
            "train": {
                "recordings": int(len(df_train)),
                "participants": int(df_train["participant_id"].nunique()),
                "healthy": int(np.sum(df_train["label"] == 0)),
                "abnormal": int(np.sum(df_train["label"] == 1))
            },
            "validation": {
                "recordings": int(len(df_val)),
                "participants": int(df_val["participant_id"].nunique()),
                "healthy": int(np.sum(df_val["label"] == 0)),
                "abnormal": int(np.sum(df_val["label"] == 1))
            },
            "test": {
                "recordings": int(len(df_test)),
                "participants": int(df_test["participant_id"].nunique()),
                "healthy": int(np.sum(df_test["label"] == 0)),
                "abnormal": int(np.sum(df_test["label"] == 1))
            }
        },
        "num_features": 91
    }

    summary_path = DATA_PROCESSED_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 4. Affichage des statistiques finales (Exigé par la section 5 du prompt)
    print("\n==================================================================")
    print(" Statistiques Finales du Jeu de Données Préparé (Anti Data-Leakage)")
    print("==================================================================")
    print(f"Total Enregistrements : {summary['total_recordings']}")
    print(f"Total Participants   : {summary['total_participants']}")
    print(f"  - Sain / Normal     : {summary['class_distribution']['Healthy']} enregistrements")
    print(f"  - Anomalie Respir.  : {summary['class_distribution']['Respiratory_Abnormality']} enregistrements")
    print("------------------------------------------------------------------")
    print(f"Train Set      : {summary['splits']['train']['recordings']} audios ({summary['splits']['train']['participants']} participants) [Sain: {summary['splits']['train']['healthy']}, Anomalie: {summary['splits']['train']['abnormal']}]")
    print(f"Validation Set : {summary['splits']['validation']['recordings']} audios ({summary['splits']['validation']['participants']} participants) [Sain: {summary['splits']['validation']['healthy']}, Anomalie: {summary['splits']['validation']['abnormal']}]")
    print(f"Test Set       : {summary['splits']['test']['recordings']} audios ({summary['splits']['test']['participants']} participants) [Sain: {summary['splits']['test']['healthy']}, Anomalie: {summary['splits']['test']['abnormal']}]")
    print("==================================================================")
    print(f"Fichiers enregistrés dans : {DATA_PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    main()
