"""
Module de génération des visualisations de Traitement du Signal pour CoughSense.

Génère des graphiques haute résolution scientifiques :
1. Forme d'onde brute vs prétraitée avec enveloppe et seuils de segmentation
2. Spectre de Fourier FFT (Amplitude & Densité spectrale de puissance avec centroïde et rolloff)
3. Spectrogramme STFT 2D (Temps-Fréquence-Puissance en dB)
4. Spectrogramme sur l'échelle de Mel
5. Matrice des 13 coefficients MFCC
6. Matrice de confusion et comparaison des modèles
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Mode sans tête pour Streamlit
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.cough_detection import CoughSegmentResult
from src.features import FeatureExtractionResult
from src.preprocessing import PreprocessingResult
from src.signal_analysis import FrequencyDomainAnalysis, TimeDomainAnalysis, TimeFrequencyAnalysis


# Configuration esthétique sobre et académique
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {
    "primary": "#1e3a8a",       # Bleu marine académique
    "secondary": "#0284c7",     # Bleu ciel profond
    "accent": "#0f766e",        # Sarcelle sobre
    "warning": "#d97706",       # Ambre sobre
    "danger": "#b91c1c",        # Rouge sobre
    "neutral_dark": "#334155",  # Ardoise foncée
    "neutral_light": "#f8fafc", # Blanc cassé
    "grid": "#e2e8f0"           # Gris très clair
}


def plot_waveform_and_segmentation(
    raw_signal: np.ndarray,
    prep_result: PreprocessingResult,
    segment_result: CoughSegmentResult,
    sample_rate: int = 16000
) -> plt.Figure:
    """
    Trace la forme d'onde temporelle brute et prétraitée avec l'enveloppe d'énergie
    et les bornes temporelles de la toux détectée.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True, dpi=120)

    t_raw = np.linspace(0, len(raw_signal) / sample_rate, len(raw_signal), endpoint=False)
    t_prep = np.linspace(0, len(prep_result.processed_signal) / sample_rate, len(prep_result.processed_signal), endpoint=False)

    # 1. Signal brut
    ax1.plot(t_raw, raw_signal, color="#64748b", alpha=0.8, linewidth=0.8, label="Signal standardisé (16 kHz)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Forme d'onde brute vs filtrée (Butterworth 80-4000 Hz)", fontsize=10, fontweight="bold", color=PALETTE["neutral_dark"])
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])

    # 2. Signal prétraité et enveloppe
    ax2.plot(t_prep, prep_result.processed_signal, color=PALETTE["primary"], linewidth=0.85, label="Signal filtré & normalisé")

    if segment_result.energy_envelope is not None and len(segment_result.energy_envelope) > 1:
        ax2.plot(
            segment_result.envelope_times,
            segment_result.energy_envelope,
            color=PALETTE["warning"],
            linewidth=1.5,
            label="Enveloppe RMS"
        )
        ax2.axhline(
            segment_result.threshold,
            color=PALETTE["danger"],
            linestyle="--",
            linewidth=1.1,
            label=f"Seuil adaptatif ({segment_result.threshold:.3f})"
        )

    # Zone de segmentation de la toux
    if segment_result.cough_detected:
        ax2.axvspan(
            segment_result.start_time,
            segment_result.end_time,
            color="#10b981",
            alpha=0.22,
            label=f"Segment toux [{segment_result.start_time:.2f}s - {segment_result.end_time:.2f}s]"
        )

    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])

    plt.tight_layout()
    return fig


def plot_frequency_spectrum(
    freq_analysis: FrequencyDomainAnalysis,
    max_freq_display: float = 6000.0
) -> plt.Figure:
    """
    Génère le tracé du spectre de Fourier (FFT) : spectre d'amplitude et DSP
    avec mise en évidence de la fréquence dominante, du centroïde spectral et du rolloff.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.8), dpi=120)

    mask = freq_analysis.frequencies <= max_freq_display
    freqs = freq_analysis.frequencies[mask]
    mag = freq_analysis.magnitude_spectrum[mask]
    power = freq_analysis.power_spectrum[mask]

    # 1. Spectre d'amplitude linéaire
    ax1.plot(freqs, mag, color=PALETTE["primary"], linewidth=1.1, label="|X(f)| (Amplitude)")
    ax1.axvline(
        freq_analysis.dominant_frequency_hz,
        color=PALETTE["danger"],
        linestyle="--",
        linewidth=1.3,
        label=f"Pic dominant : {freq_analysis.dominant_frequency_hz:.1f} Hz"
    )
    ax1.axvline(
        freq_analysis.spectral_centroid_hz,
        color=PALETTE["warning"],
        linestyle="-.",
        linewidth=1.3,
        label=f"Centroïde : {freq_analysis.spectral_centroid_hz:.1f} Hz"
    )
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Spectre de Fourier (FFT)", fontsize=10, fontweight="bold", color=PALETTE["neutral_dark"])
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])

    # 2. Densité Spectrale de Puissance (échelle logarithmique / dB)
    power_db = 10.0 * np.log10(power + 1e-12)
    ax2.plot(freqs, power_db, color=PALETTE["accent"], linewidth=1.0, label="DSP (dB/Hz)")
    ax2.axvline(
        freq_analysis.spectral_rolloff_hz,
        color="#059669",
        linestyle=":",
        linewidth=1.3,
        label=f"Rolloff (85%) : {freq_analysis.spectral_rolloff_hz:.1f} Hz"
    )
    ax2.set_xlabel("Fréquence (Hz)")
    ax2.set_ylabel("DSP (dB/Hz)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])

    plt.tight_layout()
    return fig


def plot_time_frequency_representations(
    tf_analysis: TimeFrequencyAnalysis,
    max_freq_hz: float = 6000.0
) -> plt.Figure:
    """
    Génère côte à côte le spectrogramme STFT et le Mel-Spectrogramme en dB.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), dpi=120)

    # 1. Spectrogramme STFT
    freq_mask = tf_analysis.stft_freqs <= max_freq_hz
    stft_sub = tf_analysis.spectrogram_db[freq_mask, :]
    freqs_sub = tf_analysis.stft_freqs[freq_mask]

    im1 = ax1.pcolormesh(
        tf_analysis.stft_times,
        freqs_sub,
        stft_sub,
        shading="gouraud",
        cmap="magma"
    )
    ax1.set_title(f"Spectrogramme STFT (N_FFT={tf_analysis.n_fft}, Hop={tf_analysis.hop_length})", fontsize=10, fontweight="bold", color=PALETTE["neutral_dark"])
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Fréquence (Hz)")
    cbar1 = fig.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label("Puissance (dB)", fontsize=8)

    # 2. Mel-Spectrogramme
    im2 = ax2.pcolormesh(
        tf_analysis.stft_times,
        tf_analysis.mel_freqs,
        tf_analysis.mel_spectrogram_db,
        shading="gouraud",
        cmap="viridis"
    )
    ax2.set_title("Mel-Spectrogramme (64 bandes)", fontsize=10, fontweight="bold", color=PALETTE["neutral_dark"])
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Fréquence Mel (Hz)")
    cbar2 = fig.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label("Énergie Mel (dB)", fontsize=8)

    plt.tight_layout()
    return fig


def plot_mfcc_matrix(
    mfcc_result: FeatureExtractionResult
) -> plt.Figure:
    """
    Affiche la matrice 2D des 13 coefficients cepstraux MFCC à travers le temps.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120)

    n_mfcc = mfcc_result.mfcc_matrix.shape[0]
    im = ax.pcolormesh(
        mfcc_result.mfcc_times,
        np.arange(1, n_mfcc + 1),
        mfcc_result.mfcc_matrix,
        shading="auto",
        cmap="coolwarm"
    )

    ax.set_title("Coefficients Cepstraux MFCC (1 à 13)", fontsize=10, fontweight="bold", color=PALETTE["neutral_dark"])
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Coefficient MFCC")
    ax.set_yticks(np.arange(1, n_mfcc + 1, 2))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Amplitude", fontsize=8)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str] = ["Sain / Normal", "Anomalie Respiratoire"]
) -> plt.Figure:
    """Affiche la matrice de confusion normalisée et brute."""
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=120)

    cm_array = np.array(cm)
    cm_norm = cm_array.astype("float") / (cm_array.sum(axis=1)[:, np.newaxis] + 1e-12)

    annot = np.empty_like(cm_array, dtype=object)
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            annot[i, j] = f"{cm_array[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#cbd5e1"
    )

    ax.set_title("Matrice de Confusion (Held-Out Test)", fontsize=10, fontweight="bold", color=PALETTE["neutral_dark"])
    ax.set_xlabel("Prédiction", fontsize=9)
    ax.set_ylabel("Vérité terrain", fontsize=9)

    plt.tight_layout()
    return fig


def plot_model_comparison_bars(
    metrics_dict: Dict[str, dict]
) -> plt.Figure:
    """
    Génère un graphique à barres sobres comparant Accuracy et F1-Score des modèles réels.
    """
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=120)

    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m].get("accuracy", 0.0) * 100 for m in models]
    f1_scores = [metrics_dict[m].get("f1_macro", 0.0) * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, accuracies, width, label="Accuracy (%)", color="#1e3a8a", alpha=0.85)
    rects2 = ax.bar(x + width/2, f1_scores, width, label="Score F1 Macro (%)", color="#0284c7", alpha=0.85)

    ax.set_ylabel("Performance (%)", fontsize=9)
    ax.set_title("Comparaison des Modèles sur Test Set Indépendant", fontsize=10, fontweight="bold", color=PALETTE["neutral_dark"])
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Valeurs textuelles au-dessus des barres
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    return fig
