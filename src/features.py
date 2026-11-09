"""
Module d'extraction des caractéristiques acoustiques et cepstrales (MFCC) pour CoughSense.

Construit un vecteur de caractéristiques multidimensionnel combinant :
- Descripteurs temporels (RMS, ZCR, pic, durée, facteur de crête, statistiques d'enveloppe)
- Descripteurs spectraux (centroïde, bande passante, rolloff, entropie spectrale, platitude, pic dominant)
- Descripteurs cepstraux (13 coefficients MFCC, statistiques de moyennes et écart-types, deltas et delta-deltas)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd

from src.signal_analysis import (
    analyze_frequency_domain,
    analyze_time_domain,
    FrequencyDomainAnalysis,
    TimeDomainAnalysis,
)


@dataclass
class FeatureExtractionResult:
    """Résultat de l'extraction de l'ensemble des caractéristiques du signal."""
    feature_vector: np.ndarray             # Vecteur 1D numérique
    feature_dict: Dict[str, float]         # Dictionnaire {nom: valeur}
    feature_dataframe: pd.DataFrame        # DataFrame Pandas (1 ligne x N colonnes)
    feature_names: List[str]               # Liste ordonnée des noms de caractéristiques
    mfcc_matrix: np.ndarray                # Matrice MFCC complète (13 x trames)
    delta_mfcc_matrix: np.ndarray          # Matrice Delta MFCC
    delta2_mfcc_matrix: np.ndarray         # Matrice Delta-Delta MFCC
    mfcc_times: np.ndarray                 # Axe temporel des trames MFCC


def compute_mfcc_features(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule la matrice des coefficients cepstraux sur échelle de Mel (MFCC),
    ainsi que leurs dérivées temporelles du premier ordre (Delta) et du second ordre (Delta-Delta).

    Formulation théorique :
    1. Calcul du banc de filtres de Mel : S(m)
    2. Logarithme des énergies de sous-bandes : log(S(m))
    3. Transformée en Cosinus Discrète de type II (DCT-II) :
       c_n = sum_{m=0}^{M-1} log(S(m)) * cos( pi * n * (m + 0.5) / M )

    Args:
        signal: Signal audio 1D.
        sample_rate: Fréquence d'échantillonnage en Hz.
        n_mfcc: Nombre de coefficients cepstraux (13 par défaut).
        n_fft: Taille FFT.
        hop_length: Pas d'avancement des trames.
        n_mels: Nombre de filtres Mel.

    Returns:
        Tuple[mfcc, delta_mfcc, delta2_mfcc, times]: Matrices des coefficients et axe temporel.
    """
    sig = np.asarray(signal, dtype=np.float32)

    # Calcul des MFCC
    mfcc = librosa.feature.mfcc(
        y=sig,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Dérivées temporelles (vitesses et accélérations acoustiques)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sample_rate, hop_length=hop_length)

    return mfcc, delta_mfcc, delta2_mfcc, times


def extract_features(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 128,
    include_deltas: bool = True
) -> FeatureExtractionResult:
    """
    Extrait l'ensemble complet des caractéristiques physiques et statistiques du signal de toux.

    Construit un vecteur de caractéristiques structuré et reproductible pour l'entraînement ML.

    Args:
        signal: Signal audio prétraité (1D float32).
        sample_rate: Fréquence d'échantillonnage (16000 Hz).
        n_mfcc: Nombre de coefficients MFCC extraits.
        n_fft: Taille de la fenêtre FFT.
        hop_length: Pas temporel.
        include_deltas: Si Vrai, ajoute les statistiques des dérivées Delta et Delta-Delta.

    Returns:
        FeatureExtractionResult: Conteneur complet des descripteurs numériques.
    """
    features: Dict[str, float] = {}

    # 1. Caractéristiques temporelles
    t_analysis: TimeDomainAnalysis = analyze_time_domain(signal, sample_rate=sample_rate)
    features["duration_sec"] = float(t_analysis.duration_sec)
    features["peak_amplitude"] = float(t_analysis.peak_amplitude)
    features["rms_energy"] = float(t_analysis.rms_energy)
    features["zero_crossing_rate"] = float(t_analysis.zero_crossing_rate)
    features["crest_factor"] = float(t_analysis.crest_factor)
    features["envelope_mean"] = float(np.mean(t_analysis.energy_envelope))
    features["envelope_std"] = float(np.std(t_analysis.energy_envelope))

    # 2. Caractéristiques fréquentielles globales (FFT)
    f_analysis: FrequencyDomainAnalysis = analyze_frequency_domain(signal, sample_rate=sample_rate)
    features["dominant_frequency_hz"] = float(f_analysis.dominant_frequency_hz)
    features["spectral_centroid_hz"] = float(f_analysis.spectral_centroid_hz)
    features["spectral_bandwidth_hz"] = float(f_analysis.spectral_bandwidth_hz)
    features["spectral_rolloff_hz"] = float(f_analysis.spectral_rolloff_hz)
    features["spectral_entropy"] = float(f_analysis.spectral_entropy)
    features["spectral_flatness"] = float(f_analysis.spectral_flatness)

    # 3. Caractéristiques cepstrales (MFCC)
    mfcc, delta_mfcc, delta2_mfcc, times = compute_mfcc_features(
        signal,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Moyennes et écart-types des coefficients MFCC statiques
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i, :]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i, :]))

    # Statistiques des dérivées temporelles si demandées
    if include_deltas:
        for i in range(n_mfcc):
            features[f"delta_mfcc_{i+1}_mean"] = float(np.mean(delta_mfcc[i, :]))
            features[f"delta_mfcc_{i+1}_std"] = float(np.std(delta_mfcc[i, :]))
        for i in range(n_mfcc):
            features[f"delta2_mfcc_{i+1}_mean"] = float(np.mean(delta2_mfcc[i, :]))
            features[f"delta2_mfcc_{i+1}_std"] = float(np.std(delta2_mfcc[i, :]))

    # Construction des représentations ordonnées
    feature_names = list(features.keys())
    feature_vector = np.array([features[k] for k in feature_names], dtype=np.float32)
    feature_dataframe = pd.DataFrame([features])

    return FeatureExtractionResult(
        feature_vector=feature_vector,
        feature_dict=features,
        feature_dataframe=feature_dataframe,
        feature_names=feature_names,
        mfcc_matrix=mfcc,
        delta_mfcc_matrix=delta_mfcc,
        delta2_mfcc_matrix=delta2_mfcc,
        mfcc_times=times
    )
