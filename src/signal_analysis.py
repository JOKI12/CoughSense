"""
Module d'analyse temporelle, fréquentielle (FFT) et temps-fréquence (STFT, Spectrogramme)
pour CoughSense.

Implémente rigoureusement les formulations théoriques du Traitement du Signal :
- Temporel : RMS, Crête, Durée, Taux de passage par zéro (ZCR), Facteur de crête, Enveloppe
- Fréquentiel : FFT réelle, Spectre d'amplitude, Spectre de puissance, Fréquence dominante,
  Centroïde spectral, Largeur de bande spectrale, Rolloff, Entropie spectrale, Platitude spectrale
- Temps-Fréquence : STFT, Spectrogramme de puissance (dB), Spectrogramme de Mel
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import numpy as np
from scipy import signal as sp_signal


@dataclass
class TimeDomainAnalysis:
    """Métriques et caractéristiques de l'analyse dans le domaine temporel."""
    duration_sec: float
    peak_amplitude: float
    rms_energy: float
    zero_crossing_rate: float
    crest_factor: float
    energy_envelope: np.ndarray
    envelope_times: np.ndarray


@dataclass
class FrequencyDomainAnalysis:
    """Métriques et spectres de l'analyse dans le domaine fréquentiel (FFT)."""
    frequencies: np.ndarray          # Axe des fréquences physiques en Hz [0, fs/2]
    magnitude_spectrum: np.ndarray   # Spectre d'amplitude normalisé
    power_spectrum: np.ndarray       # Densité spectrale de puissance (DSP)
    dominant_frequency_hz: float     # Fréquence présentant le pic d'énergie maximal
    spectral_centroid_hz: float      # Barycentre spectral (centre de gravité des fréquences)
    spectral_bandwidth_hz: float     # Dispersion / écart-type spectral
    spectral_rolloff_hz: float       # Fréquence sous laquelle réside 85% de l'énergie
    spectral_entropy: float          # Entropie spectrale normalisée (degré de désordre spectral)
    spectral_flatness: float         # Platitude spectrale (tonalité vs bruit blanc)


@dataclass
class TimeFrequencyAnalysis:
    """Représentations temps-fréquence (STFT et Spectrogrammes)."""
    stft_times: np.ndarray           # Axe temporel des trames (s)
    stft_freqs: np.ndarray           # Axe fréquentiel de la STFT (Hz)
    spectrogram_db: np.ndarray       # Spectrogramme de puissance en dB (freqs x times)
    mel_freqs: np.ndarray            # Axe fréquentiel de l'échelle de Mel (Hz)
    mel_spectrogram_db: np.ndarray   # Spectrogramme de Mel en dB (mels x times)
    n_fft: int
    hop_length: int
    win_length: int
    window_name: str


@dataclass
class SignalAnalysisResult:
    """Conteneur global de toutes les analyses de signal processing."""
    time_domain: TimeDomainAnalysis
    frequency_domain: FrequencyDomainAnalysis
    time_frequency: TimeFrequencyAnalysis


def analyze_time_domain(
    signal: np.ndarray,
    sample_rate: int = 16000
) -> TimeDomainAnalysis:
    """
    Calcule les métriques temporelles standardisées d'un signal audio.

    Formulations :
    - Durée : T = N / fs
    - Pic : max(|x[n]|)
    - RMS : sqrt( (1 / N) * sum(x[n]^2) )
    - ZCR : (1 / (2 * (N - 1))) * sum( |sgn(x[n]) - sgn(x[n - 1])| )
    - Facteur de crête : Pic / RMS

    Args:
        signal: Signal 1D.
        sample_rate: Fréquence d'échantillonnage en Hz.

    Returns:
        TimeDomainAnalysis: Structure des métriques temporelles.
    """
    sig = np.asarray(signal, dtype=np.float32)
    n_samples = len(sig)
    duration = float(n_samples / float(sample_rate))

    peak = float(np.max(np.abs(sig))) if n_samples > 0 else 0.0
    rms = float(np.sqrt(np.mean(sig**2))) if n_samples > 0 else 0.0

    # Taux de passage par zéro
    if n_samples > 1:
        zero_crossings = np.sum(np.abs(np.diff(np.sign(sig) >= 0)))
        zcr = float(zero_crossings / float(2 * (n_samples - 1)))
    else:
        zcr = 0.0

    # Facteur de crête
    crest_factor = float(peak / (rms + 1e-12))

    # Enveloppe d'énergie par transformée de Hilbert
    analytic_signal = sp_signal.hilbert(sig)
    envelope = np.abs(analytic_signal)
    envelope_times = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)

    return TimeDomainAnalysis(
        duration_sec=duration,
        peak_amplitude=peak,
        rms_energy=rms,
        zero_crossing_rate=zcr,
        crest_factor=crest_factor,
        energy_envelope=np.asarray(envelope, dtype=np.float32),
        envelope_times=envelope_times
    )


def analyze_frequency_domain(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_fft: Optional[int] = None,
    window_name: str = "hann",
    rolloff_percent: float = 0.85
) -> FrequencyDomainAnalysis:
    """
    Exécute la Transformée de Fourier Rapide (FFT) et extrait les descripteurs spectraux.

    Formulations :
    - X[k] = sum_{n=0}^{N-1} x[n] w[n] e^{-j 2 pi k n / N}
    - Spectre d'amplitude : A[k] = (2 / sum(w)) * |X[k]|
    - DSP : P[k] = (1 / (fs * sum(w^2))) * |X[k]|^2
    - Centroïde spectral : f_c = sum( f[k] * A[k] ) / sum( A[k] )
    - Largeur de bande spectrale : sigma_f = sqrt( sum( (f[k] - f_c)^2 * A[k] ) / sum( A[k] ) )
    - Entropie spectrale : H = - sum( p[k] * log2(p[k]) ) / log2(K)

    Args:
        signal: Signal audio 1D.
        sample_rate: Fréquence d'échantillonnage en Hz.
        n_fft: Taille FFT (si None, puissance de 2 immédiatement supérieure à N).
        window_name: Fenêtre de pondération temporelle ('hann', 'hamming', 'blackman', 'boxcar').
        rolloff_percent: Pourcentage cumulé d'énergie pour le calcul du rolloff (défaut : 85%).

    Returns:
        FrequencyDomainAnalysis: Conteneur complet des descripteurs spectraux.
    """
    sig = np.asarray(signal, dtype=np.float64)
    n_samples = len(sig)

    # Assurer que n_fft couvre l'intégralité du signal (puissance de 2 avec zero-padding)
    next_pow2 = max(512, int(2 ** np.ceil(np.log2(max(n_samples, 512)))))
    if n_fft is None or n_fft < n_samples:
        n_fft = next_pow2

    # Fenêtrage temporel
    if window_name == "hann":
        window = np.hanning(n_samples)
    elif window_name == "hamming":
        window = np.hamming(n_samples)
    elif window_name == "blackman":
        window = np.blackman(n_samples)
    else:
        window = np.ones(n_samples)

    windowed_signal = sig * window

    # Calcul de la FFT réelle
    fft_complex = np.fft.rfft(windowed_signal, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # Spectre d'amplitude normalisé
    win_sum = np.sum(window) + 1e-12
    mag_spectrum = (2.0 / win_sum) * np.abs(fft_complex)
    mag_spectrum[0] /= 2.0  # Correction composante continue DC

    # Densité spectrale de puissance (DSP)
    win_sq_sum = np.sum(window**2) + 1e-12
    power_spectrum = (1.0 / (sample_rate * win_sq_sum)) * (np.abs(fft_complex) ** 2)

    # Fréquence dominante (pic spectral hors DC)
    valid_idx = np.arange(1, len(mag_spectrum))
    dom_idx = valid_idx[np.argmax(mag_spectrum[valid_idx])] if len(valid_idx) > 0 else 0
    dominant_freq = float(freqs[dom_idx])

    # Centroïde spectral
    total_mag = np.sum(mag_spectrum) + 1e-12
    spectral_centroid = float(np.sum(freqs * mag_spectrum) / total_mag)

    # Largeur de bande spectrale (écart-type spectral)
    spectral_bandwidth = float(
        np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * mag_spectrum) / total_mag)
    )

    # Rolloff spectral (fréquence seuil à 85% d'énergie cumulée)
    cum_power = np.cumsum(power_spectrum)
    total_power = cum_power[-1] + 1e-12
    rolloff_idx = np.searchsorted(cum_power, rolloff_percent * total_power)
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    spectral_rolloff = float(freqs[rolloff_idx])

    # Entropie spectrale (mesure du désordre / stochasticité fréquentielle)
    norm_power = power_spectrum / total_power
    norm_power = norm_power[norm_power > 0]
    n_bins = len(power_spectrum)
    raw_entropy = -np.sum(norm_power * np.log2(norm_power + 1e-12))
    max_entropy = np.log2(n_bins) if n_bins > 1 else 1.0
    spectral_entropy = float(np.clip(raw_entropy / max_entropy, 0.0, 1.0))

    # Platitude spectrale (Spectral Flatness = moyenne géométrique / moyenne arithmétique)
    pos_power = power_spectrum[power_spectrum > 1e-15]
    if len(pos_power) > 0:
        geom_mean = np.exp(np.mean(np.log(pos_power)))
        arith_mean = np.mean(pos_power)
        spectral_flatness = float(np.clip(geom_mean / (arith_mean + 1e-12), 0.0, 1.0))
    else:
        spectral_flatness = 0.0

    return FrequencyDomainAnalysis(
        frequencies=np.asarray(freqs, dtype=np.float32),
        magnitude_spectrum=np.asarray(mag_spectrum, dtype=np.float32),
        power_spectrum=np.asarray(power_spectrum, dtype=np.float32),
        dominant_frequency_hz=dominant_freq,
        spectral_centroid_hz=spectral_centroid,
        spectral_bandwidth_hz=spectral_bandwidth,
        spectral_rolloff_hz=spectral_rolloff,
        spectral_entropy=spectral_entropy,
        spectral_flatness=spectral_flatness
    )


def analyze_time_frequency(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 128,
    window_name: str = "hann",
    n_mels: int = 64,
    f_min: float = 50.0,
    f_max: float = 7500.0
) -> TimeFrequencyAnalysis:
    """
    Calcule la STFT, le Spectrogramme de puissance (dB) et le Mel-Spectrogramme (dB).

    Formulation STFT :
    X(m, k) = sum_{n=0}^{N_w - 1} x[n + m * H] w[n] e^{-j 2 pi k n / N_{FFT}}

    Formulation Mel :
    m = 2595 * log10(1 + f / 700)

    Args:
        signal: Signal audio 1D.
        sample_rate: Fréquence d'échantillonnage en Hz.
        n_fft: Nombre de points FFT.
        win_length: Longueur de la fenêtre d'observation temporelle.
        hop_length: Pas d'avancement temporel entre trames successives.
        window_name: Type de fenêtre de troncature.
        n_mels: Nombre de bandes de filtres triangulaires sur l'échelle de Mel.
        f_min: Fréquence minimale d'analyse Mel.
        f_max: Fréquence maximale d'analyse Mel.

    Returns:
        TimeFrequencyAnalysis: Matrices 2D et axes coordonnés pour le rendu graphique.
    """
    sig = np.asarray(signal, dtype=np.float32)

    # 1. Calcul de la STFT
    stft_matrix = librosa.stft(
        sig,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_name
    )

    stft_mag = np.abs(stft_matrix)
    stft_power = stft_mag ** 2

    # Conversion en échelle logarithmique (dB relatif au maximum)
    spectrogram_db = librosa.power_to_db(stft_power, ref=np.max, top_db=80.0)

    stft_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    stft_times = librosa.frames_to_time(np.arange(stft_matrix.shape[1]), sr=sample_rate, hop_length=hop_length)

    # 2. Calcul du Mel-Spectrogramme
    mel_power = librosa.feature.melspectrogram(
        y=sig,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_name,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max
    )
    mel_spectrogram_db = librosa.power_to_db(mel_power, ref=np.max, top_db=80.0)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=f_min, fmax=f_max)

    return TimeFrequencyAnalysis(
        stft_times=np.asarray(stft_times, dtype=np.float32),
        stft_freqs=np.asarray(stft_freqs, dtype=np.float32),
        spectrogram_db=np.asarray(spectrogram_db, dtype=np.float32),
        mel_freqs=np.asarray(mel_freqs, dtype=np.float32),
        mel_spectrogram_db=np.asarray(mel_spectrogram_db, dtype=np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_name=window_name
    )


def analyze_signal(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_fft_spectrum: Optional[int] = 1024,
    stft_n_fft: int = 512,
    stft_win_length: int = 512,
    stft_hop_length: int = 128,
    stft_window: str = "hann",
    n_mels: int = 64
) -> SignalAnalysisResult:
    """
    Exécute conjointement les analyses temporelle, fréquentielle et temps-fréquence.
    """
    t_res = analyze_time_domain(signal, sample_rate=sample_rate)
    f_res = analyze_frequency_domain(
        signal,
        sample_rate=sample_rate,
        n_fft=n_fft_spectrum,
        window_name=stft_window
    )
    tf_res = analyze_time_frequency(
        signal,
        sample_rate=sample_rate,
        n_fft=stft_n_fft,
        win_length=stft_win_length,
        hop_length=stft_hop_length,
        window_name=stft_window,
        n_mels=n_mels
    )

    return SignalAnalysisResult(
        time_domain=t_res,
        frequency_domain=f_res,
        time_frequency=tf_res
    )
