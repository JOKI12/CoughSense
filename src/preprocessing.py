"""
Module de prétraitement du signal audio pour CoughSense.

Implémente la chaîne complète de conditionnement du signal :
1. Normalisation d'amplitude crête / RMS
2. Débruitage par soustraction spectrale (spectral gating)
3. Filtrage passe-bande Butterworth à phase nulle (sosfiltfilt)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import numpy as np
from scipy import signal as sp_signal


@dataclass
class PreprocessingResult:
    """Résultat complet des étapes de prétraitement du signal."""
    processed_signal: np.ndarray      # Signal filtré, débruité et normalisé
    filtered_signal: np.ndarray       # Signal après filtrage passe-bande
    denoised_signal: np.ndarray       # Signal après réduction de bruit
    normalized_signal: np.ndarray     # Signal après normalisation
    sample_rate: int                  # Fréquence d'échantillonnage
    snr_estimated_db: float           # Rapport signal-sur-bruit estimé (dB)
    noise_reduction_applied: bool     # Indicateur de débruitage effectif
    filter_bandwidth: Tuple[float, float]  # Bande passante appliquée [f_low, f_high]


def normalize_signal(
    signal: np.ndarray,
    target_peak: float = 0.95,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalise l'amplitude crête d'un signal audio numérique.

    x_norm[n] = (x[n] / (max(|x|) + eps)) * target_peak

    Args:
        signal: Tableau 1D d'échantillons audio.
        target_peak: Valeur crête maximale visée dans [-1.0, 1.0] (défaut : 0.95 pour éviter l'écrêtage).
        eps: Facteur de stabilisation numérique contre la division par zéro.

    Returns:
        np.ndarray: Signal normalisé en float32.
    """
    signal = np.asarray(signal, dtype=np.float32)
    current_peak = np.max(np.abs(signal))
    if current_peak < eps:
        return signal.copy()
    scaling_factor = target_peak / (current_peak + eps)
    return np.asarray(signal * scaling_factor, dtype=np.float32)


def apply_spectral_subtraction(
    signal: np.ndarray,
    sample_rate: int,
    noise_frames_ratio: float = 0.10,
    oversubtraction_factor: float = 1.3,
    spectral_floor: float = 0.05,
    n_fft: int = 512,
    hop_length: int = 128
) -> Tuple[np.ndarray, float]:
    """
    Réduit le bruit de fond stationnaire par soustraction spectrale non-linéaire.

    Principe de traitement du signal :
    1. Calcul de la STFT (Short-Time Fourier Transform) du signal bruité : X(m, k)
    2. Estimation du spectre de puissance du bruit N(k) à partir des trames les plus silencieuses (énergie minimale)
    3. Soustraction spectrale : |S_est(m, k)| = max(|X(m, k)| - alpha * |N(k)|, beta * |X(m, k)|)
    4. Reconstruction temporelle par ISTFT (Inverse STFT) en conservant la phase originale :
       s_est[n] = ISTFT(|S_est(m, k)| * exp(j * angle(X(m, k))))

    Args:
        signal: Signal audio 1D.
        sample_rate: Fréquence d'échantillonnage en Hz.
        noise_frames_ratio: Fraction des trames à plus faible énergie utilisées pour modéliser le bruit.
        oversubtraction_factor: Facteur d'alpha pour sur-soustraire le bruit sans trop altérer les formants.
        spectral_floor: Plancher spectral beta pour éliminer les artéfacts de "bruit musical".
        n_fft: Taille de la fenêtre FFT pour la STFT.
        hop_length: Pas d'avancement temporel entre trames successives.

    Returns:
        Tuple[np.ndarray, float]: (Signal débruité, SNR estimé en dB).
    """
    # Calcul de la STFT
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window="hann")
    magnitude, phase = np.abs(D), np.angle(D)
    power = magnitude ** 2

    # Calcul de l'énergie par trame pour identifier les trames de silence/bruit pur
    frame_energies = np.sum(power, axis=0)
    n_frames = power.shape[1]
    n_noise_frames = max(2, int(n_frames * noise_frames_ratio))

    # Trames à plus faible énergie (bruit de fond ambiant)
    noise_frame_indices = np.argsort(frame_energies)[:n_noise_frames]
    noise_power_profile = np.mean(power[:, noise_frame_indices], axis=1, keepdims=True)
    noise_mag_profile = np.sqrt(noise_power_profile)

    # Estimation du SNR brut
    signal_power_mean = np.mean(power)
    noise_power_mean = np.mean(noise_power_profile)
    snr_db = float(10.0 * np.log10(max(signal_power_mean / (noise_power_mean + 1e-12), 1e-4)))

    # Soustraction spectrale
    subtracted_mag = magnitude - oversubtraction_factor * noise_mag_profile
    floor_mag = spectral_floor * magnitude
    cleaned_mag = np.maximum(subtracted_mag, floor_mag)

    # Synthèse du signal débruité par ISTFT
    cleaned_stft = cleaned_mag * np.exp(1j * phase)
    denoised = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(signal), window="hann")

    return np.asarray(denoised, dtype=np.float32), snr_db


def apply_bandpass_filter(
    signal: np.ndarray,
    sample_rate: int,
    low_cutoff_hz: float = 80.0,
    high_cutoff_hz: float = 4000.0,
    order: int = 4
) -> np.ndarray:
    """
    Applique un filtre numérique passe-bande Butterworth à phase nulle (bidirectionnel).

    Justification scientifique acoustique :
    - F < 80 Hz : Bruits mécaniques, frottements de microphone, bruits de pas, résonances cardiaques.
    - F > 4000 Hz : Bruits thermiques haute fréquence et sifflements hors de la zone spectrale utile de la toux.
    - L'utilisation de SOS (Second-Order Sections) garantit la stabilité numérique.
    - L'application bidirectionnelle (sosfiltfilt) élimine toute distorsion de phase (phase linéaire stricte nulle).

    Args:
        signal: Signal audio 1D.
        sample_rate: Fréquence d'échantillonnage en Hz.
        low_cutoff_hz: Fréquence de coupure basse en Hz (défaut : 80 Hz).
        high_cutoff_hz: Fréquence de coupure haute en Hz (défaut : 4000 Hz).
        order: Ordre du filtre Butterworth (défaut : 4, fournissant une atténuation de 48 dB/octave).

    Returns:
        np.ndarray: Signal filtré.
    """
    nyquist = 0.5 * sample_rate

    # Vérification des limites de Nyquist
    low = max(10.0, low_cutoff_hz) / nyquist
    high = min(nyquist - 10.0, high_cutoff_hz) / nyquist

    if low >= high:
        raise ValueError(
            f"Paramètres de coupure invalides : low={low_cutoff_hz} Hz, high={high_cutoff_hz} Hz "
            f"pour fs={sample_rate} Hz (Nyquist={nyquist} Hz)."
        )

    # Synthèse du filtre sous forme SOS (Second-Order Sections)
    sos = sp_signal.butter(order, [low, high], btype="bandpass", output="sos")

    # Filtrage à phase nulle
    filtered = sp_signal.sosfiltfilt(sos, signal)
    return np.asarray(filtered, dtype=np.float32)


def preprocess_signal(
    signal: np.ndarray,
    sample_rate: int = 16000,
    apply_noise_reduction: bool = True,
    apply_filtering: bool = True,
    low_cutoff_hz: float = 80.0,
    high_cutoff_hz: float = 4000.0,
    filter_order: int = 4,
    target_peak: float = 0.95
) -> PreprocessingResult:
    """
    Exécute la chaîne séquentielle complète de prétraitement du signal.

    Pipeline :
    Signal brut -> Débruitage spectral -> Filtrage passe-bande -> Normalisation d'amplitude

    Args:
        signal: Échantillons audio standardisés 1D.
        sample_rate: Fréquence d'échantillonnage (16000 Hz).
        apply_noise_reduction: Active ou désactive la réduction de bruit spectrale.
        apply_filtering: Active ou désactive le filtrage passe-bande.
        low_cutoff_hz: Fréquence basse de coupure.
        high_cutoff_hz: Fréquence haute de coupure.
        filter_order: Ordre du filtre Butterworth.
        target_peak: Amplitude crête normalisée.

    Returns:
        PreprocessingResult: Conteneur complet des signaux à chaque étape et métriques.
    """
    current_sig = np.asarray(signal, dtype=np.float32).copy()

    # 1. Débruitage spectral
    snr_db = 0.0
    denoised_sig = current_sig.copy()
    if apply_noise_reduction:
        denoised_sig, snr_db = apply_spectral_subtraction(current_sig, sample_rate=sample_rate)
        current_sig = denoised_sig.copy()

    # 2. Filtrage passe-bande
    filtered_sig = current_sig.copy()
    if apply_filtering:
        filtered_sig = apply_bandpass_filter(
            current_sig,
            sample_rate=sample_rate,
            low_cutoff_hz=low_cutoff_hz,
            high_cutoff_hz=high_cutoff_hz,
            order=filter_order
        )
        current_sig = filtered_sig.copy()

    # 3. Normalisation d'amplitude crête
    normalized_sig = normalize_signal(current_sig, target_peak=target_peak)

    return PreprocessingResult(
        processed_signal=normalized_sig,
        filtered_signal=filtered_sig,
        denoised_signal=denoised_sig,
        normalized_signal=normalized_sig,
        sample_rate=sample_rate,
        snr_estimated_db=snr_db,
        noise_reduction_applied=apply_noise_reduction,
        filter_bandwidth=(low_cutoff_hz, high_cutoff_hz)
    )
