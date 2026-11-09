"""
Module d'inférence bout-en-bout pour CoughSense.

Réutilise EXACTEMENT le pipeline de prétraitement et d'extraction de caractéristiques
utilisé lors de l'entraînement :
Audio brut -> Standardisation 16 kHz -> Débruitage spectral -> Filtre Butterworth ->
Normalisation crête -> Segmentation de toux -> Descripteurs temporels/spectro/MFCC ->
StandardScaler -> Modèle ML -> Probabilités & Visualisations.
"""

from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.audio_loader import (
    AudioData,
    AudioFormatError,
    AudioValidationError,
    load_and_standardize_audio,
)
from src.cough_detection import CoughSegmentResult, segment_cough
from src.features import FeatureExtractionResult, extract_features
from src.preprocessing import PreprocessingResult, preprocess_signal
from src.signal_analysis import SignalAnalysisResult, analyze_signal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class ModelNotTrainedError(FileNotFoundError):
    """Exception levée lorsque les artefacts du modèle entraîné sont introuvables."""
    pass


@dataclass
class InferenceResult:
    """Résultat complet d'analyse et d'inférence pour une toux."""
    predicted_class_id: int            # 0 = Healthy, 1 = Respiratory_Abnormality
    predicted_label: str               # "Pattern sain / normal" ou "Pattern évocateur d'anomalie respiratoire"
    probabilities: Dict[str, float]    # {classe: probabilité}
    confidence_score: float            # Probabilité de la classe prédite
    model_name: str                    # Nom du modèle utilisé pour l'inférence
    features_df: pd.DataFrame          # DataFrame des 91 caractéristiques extraites
    feature_dict: Dict[str, float]     # Dictionnaire des caractéristiques
    audio_data: AudioData              # Métadonnées audio d'origine
    preprocessing_result: PreprocessingResult  # Signaux aux différentes étapes de filtrage
    cough_segment: CoughSegmentResult  # Segment de toux détecté et enveloppe
    signal_analysis: SignalAnalysisResult  # Analyses temporelle, FFT, STFT, Spectrogramme, Mel
    mfcc_result: FeatureExtractionResult   # Matrices MFCC, deltas et delta-deltas
    inference_time_ms: float           # Temps d'inférence algorithmique en ms
    disclaimer: str                    # Mention d'avertissement académique légale


class CoughInferenceEngine:
    """Moteur d'inférence réutilisable et étanche."""

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.scaler = None
        self.feature_names = None
        self.loaded_models: Dict[str, object] = {}
        self.default_model_name = "best"
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Charge le scaler et la liste des caractéristiques si disponibles."""
        scaler_path = self.models_dir / "scaler.joblib"
        feat_path = self.models_dir / "feature_names.json"

        if scaler_path.exists() and feat_path.exists():
            self.scaler = joblib.load(scaler_path)
            with open(feat_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)

    def is_model_available(self, model_name: str = "best") -> bool:
        """Vérifie si le modèle spécifié est entraîné et disponible sur le disque."""
        if self.scaler is None or self.feature_names is None:
            self._load_artifacts()
            if self.scaler is None or self.feature_names is None:
                return False

        if model_name == "best":
            target_file = self.models_dir / "best_model.joblib"
        else:
            target_file = self.models_dir / f"{model_name.lower()}_model.joblib"

        return target_file.exists()

    def _get_model(self, model_name: str = "best"):
        """Charge en mémoire un modèle demandé."""
        if not self.is_model_available(model_name):
            raise ModelNotTrainedError(
                f"Le modèle '{model_name}' n'est pas encore entraîné. "
                "Veuillez exécuter le pipeline d'entraînement préalable : python training/train.py"
            )

        key = model_name.lower()
        if key not in self.loaded_models:
            if key == "best":
                model_file = self.models_dir / "best_model.joblib"
            else:
                model_file = self.models_dir / f"{key}_model.joblib"
            self.loaded_models[key] = joblib.load(model_file)

        return self.loaded_models[key]

    def predict(
        self,
        audio_source: Union[str, Path, bytes, io.BytesIO],
        model_name: str = "best",
        source_name: Optional[str] = None
    ) -> InferenceResult:
        """
        Exécute la chaîne d'inférence complète sur un fichier audio ou un enregistrement micro.

        Args:
            audio_source: Fichier, buffer ou octets audio.
            model_name: Nom du modèle ('best', 'SVM', 'Random_Forest', 'KNN').
            source_name: Nom optionnel pour affichage.

        Returns:
            InferenceResult: Objet complet contenant prédiction, probabilités, caractéristiques et signaux.
        """
        import time
        t_start = time.perf_counter()

        # 1. Chargement et standardisation (mono, 16 kHz, validation)
        audio_data = load_and_standardize_audio(
            audio_source,
            target_sample_rate=16000,
            min_duration_sec=0.20,
            max_duration_sec=15.0,
            source_name=source_name
        )

        # 2. Prétraitement rigoureux
        prep_res = preprocess_signal(
            audio_data.signal,
            sample_rate=audio_data.sample_rate,
            apply_noise_reduction=True,
            apply_filtering=True,
            low_cutoff_hz=80.0,
            high_cutoff_hz=4000.0,
            filter_order=4,
            target_peak=0.95
        )

        # 3. Détection et segmentation de la toux
        cough_seg = segment_cough(
            prep_res.processed_signal,
            sample_rate=audio_data.sample_rate,
            threshold_multiplier=1.20
        )

        # Signal pour extraction (segment de toux ou signal complet si burst court)
        eval_signal = cough_seg.segmented_signal if len(cough_seg.segmented_signal) >= 3200 else prep_res.processed_signal

        # 4. Analyses complètes de Traitement du Signal (FFT, STFT, Spectrogrammes)
        sig_analysis = analyze_signal(
            eval_signal,
            sample_rate=audio_data.sample_rate,
            n_fft_spectrum=1024,
            stft_n_fft=512,
            stft_win_length=512,
            stft_hop_length=128,
            stft_window="hann",
            n_mels=64
        )

        # 5. Extraction des 91 descripteurs acoustiques
        feat_res = extract_features(
            eval_signal,
            sample_rate=audio_data.sample_rate,
            n_mfcc=13,
            n_fft=512,
            hop_length=128,
            include_deltas=True
        )

        # 6. Chargement du modèle et prédiction
        model = self._get_model(model_name)

        # Alignement strict des colonnes sur l'ordre d'entraînement
        feature_vector_ordered = np.array(
            [feat_res.feature_dict[col] for col in self.feature_names],
            dtype=np.float32
        ).reshape(1, -1)

        # Normalisation standard
        feature_vector_scaled = self.scaler.transform(feature_vector_ordered)

        # Prédiction ML
        pred_class_id = int(model.predict(feature_vector_scaled)[0])

        if hasattr(model, "predict_proba"):
            raw_probas = model.predict_proba(feature_vector_scaled)[0]
            prob_healthy = float(raw_probas[0])
            prob_abnormal = float(raw_probas[1])
        else:
            # Fallback pour modèles sans probabilités explicites
            prob_healthy = 1.0 if pred_class_id == 0 else 0.0
            prob_abnormal = 1.0 if pred_class_id == 1 else 0.0

        confidence = prob_abnormal if pred_class_id == 1 else prob_healthy

        # Formulation terminologique académique stricte (Section 2 & 24)
        if pred_class_id == 0:
            pred_label = "Pattern sain / normal"
        else:
            pred_label = "Pattern évocateur d'anomalie respiratoire"

        probabilities = {
            "Sain / Normal": round(prob_healthy, 4),
            "Anomalie Respiratoire": round(prob_abnormal, 4)
        }

        t_end = time.perf_counter()
        inference_latency_ms = (t_end - t_start) * 1000.0

        disclaimer_text = (
            "Système de Dépistage Expérimental et Pédagogique (Recherche / Master S3).\n"
            "Ce résultat repose exclusivement sur des caractéristiques acoustiques et spectrales "
            "et ne constitue en aucun cas un diagnostic médical. "
            "Il ne doit jamais remplacer l'avis ou l'auscultation d'un médecin pneumologue."
        )

        return InferenceResult(
            predicted_class_id=pred_class_id,
            predicted_label=pred_label,
            probabilities=probabilities,
            confidence_score=confidence,
            model_name=model_name,
            features_df=feat_res.feature_dataframe,
            feature_dict=feat_res.feature_dict,
            audio_data=audio_data,
            preprocessing_result=prep_res,
            cough_segment=cough_seg,
            signal_analysis=sig_analysis,
            mfcc_result=feat_res,
            inference_time_ms=round(inference_latency_ms, 2),
            disclaimer=disclaimer_text
        )


# Instance globale pour inférence directe
_global_engine: Optional[CoughInferenceEngine] = None


def get_inference_engine() -> CoughInferenceEngine:
    """Retourne l'instance singleton du moteur d'inférence."""
    global _global_engine
    if _global_engine is None:
        _global_engine = CoughInferenceEngine()
    return _global_engine


def predict_audio_file(
    audio_source: Union[str, Path, bytes, io.BytesIO],
    model_name: str = "best"
) -> InferenceResult:
    """Point d'entrée direct pour la prédiction d'un signal audio."""
    engine = get_inference_engine()
    return engine.predict(audio_source, model_name=model_name)
