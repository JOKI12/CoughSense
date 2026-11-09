"""
Tests unitaires pour le chargement, la validation et le prétraitement audio.
"""

import io
import numpy as np
import pytest
import soundfile as sf

from src.audio_loader import (
    AudioData,
    AudioFormatError,
    AudioValidationError,
    load_and_standardize_audio,
    validate_audio_signal,
)
from src.cough_detection import compute_energy_envelope, segment_cough
from src.preprocessing import (
    apply_bandpass_filter,
    apply_spectral_subtraction,
    normalize_signal,
    preprocess_signal,
)


def create_synthetic_wav(
    duration: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 2,
    freq: float = 440.0
) -> io.BytesIO:
    """Génère un flux WAV synthétique en mémoire."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if channels == 1:
        data = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    else:
        left = 0.5 * np.sin(2 * np.pi * freq * t)
        right = 0.3 * np.sin(2 * np.pi * (freq * 1.5) * t)
        data = np.column_stack([left, right]).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV")
    buf.seek(0)
    return buf


def test_load_and_standardize_audio_stereo_resample():
    """Vérifie la conversion stéréo vers mono et le rééchantillonnage à 16 kHz."""
    buf = create_synthetic_wav(duration=1.0, sample_rate=44100, channels=2)
    audio = load_and_standardize_audio(buf, target_sample_rate=16000)

    assert audio.sample_rate == 16000
    assert audio.original_sample_rate == 44100
    assert audio.original_channels == 2
    assert audio.signal.ndim == 1
    assert len(audio.signal) == 16000
    assert 0.95 <= audio.duration <= 1.05


def test_audio_validation_too_short():
    """Vérifie le rejet d'un enregistrement trop court."""
    short_signal = np.sin(np.linspace(0, 10, 500)).astype(np.float32)
    with pytest.raises(AudioValidationError):
        validate_audio_signal(short_signal, sample_rate=16000, min_duration_sec=0.25)


def test_audio_validation_silent():
    """Vérifie le rejet d'un signal quasi-silencieux."""
    silent_signal = np.zeros(16000, dtype=np.float32)
    with pytest.raises(AudioValidationError):
        validate_audio_signal(silent_signal, sample_rate=16000)


def test_audio_validation_nan():
    """Vérifie le rejet d'un signal corrompu contenant des NaN."""
    nan_signal = np.ones(16000, dtype=np.float32)
    nan_signal[100] = np.nan
    with pytest.raises(AudioValidationError):
        validate_audio_signal(nan_signal, sample_rate=16000)


def test_peak_normalization():
    """Vérifie la normalisation de crête."""
    sig = np.array([0.1, -0.4, 0.2, -0.3], dtype=np.float32)
    norm = normalize_signal(sig, target_peak=0.95)
    assert np.isclose(np.max(np.abs(norm)), 0.95, atol=1e-5)


def test_bandpass_filter_attenuates_out_of_band():
    """Vérifie que le filtre passe-bande atténue les fréquences hors-bande (ex: 20 Hz et 6000 Hz)."""
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    # Signal composé de 30 Hz (hors bande basse), 500 Hz (dans la bande passante), et 6500 Hz (hors bande haute)
    sig_low = np.sin(2 * np.pi * 30 * t)
    sig_pass = np.sin(2 * np.pi * 500 * t)
    sig_high = np.sin(2 * np.pi * 6500 * t)
    composite = sig_low + sig_pass + sig_high

    filtered = apply_bandpass_filter(composite, sample_rate=fs, low_cutoff_hz=80.0, high_cutoff_hz=4000.0)

    # L'énergie de la composante 500 Hz doit être largement préservée, tandis que 30 Hz et 6500 Hz sont atténuées
    fft_in = np.abs(np.fft.rfft(composite))
    fft_out = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(fs, 1.0 / fs)

    idx_30 = np.argmin(np.abs(freqs - 30))
    idx_500 = np.argmin(np.abs(freqs - 500))
    idx_6500 = np.argmin(np.abs(freqs - 6500))

    # Atténuation d'au moins 20 dB hors bande
    assert fft_out[idx_30] < 0.2 * fft_in[idx_30]
    assert fft_out[idx_6500] < 0.2 * fft_in[idx_6500]
    assert fft_out[idx_500] > 0.8 * fft_in[idx_500]


def test_cough_segmentation_burst_detection():
    """Vérifie la détection temporelle correcte d'un sursaut acoustique."""
    fs = 16000
    t = np.linspace(0, 2.0, 2 * fs, endpoint=False)
    sig = np.zeros(2 * fs, dtype=np.float32)
    # Burst entre 0.6s et 1.2s
    sig[int(0.6 * fs):int(1.2 * fs)] = 0.8 * np.sin(2 * np.pi * 400 * t[:int(0.6 * fs)])

    seg = segment_cough(sig, sample_rate=fs)
    assert seg.cough_detected is True
    assert seg.num_bursts >= 1
    assert 0.45 <= seg.start_time <= 0.65
    assert 1.15 <= seg.end_time <= 1.35
