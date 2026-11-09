"""
Tests unitaires pour l'analyse spectrale (FFT, STFT, Mel) et l'extraction des caractéristiques (MFCC).
"""

import numpy as np
import pytest

from src.features import compute_mfcc_features, extract_features
from src.signal_analysis import (
    analyze_frequency_domain,
    analyze_signal,
    analyze_time_domain,
    analyze_time_frequency,
)


def test_time_domain_analysis():
    """Vérifie le calcul des métriques temporelles."""
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 100 * t).astype(np.float32)

    res = analyze_time_domain(sig, sample_rate=fs)
    assert np.isclose(res.duration_sec, 1.0, atol=1e-3)
    assert np.isclose(res.peak_amplitude, 0.5, atol=1e-2)
    # RMS d'une sinusoïde pure d'amplitude A = A / sqrt(2) = 0.5 / 1.414 = 0.3535
    assert np.isclose(res.rms_energy, 0.5 / np.sqrt(2), atol=1e-2)
    assert res.zero_crossing_rate > 0.0
    assert np.isclose(res.crest_factor, np.sqrt(2), atol=0.1)


def test_fft_dominant_frequency_detection():
    """Vérifie que la FFT identifie avec précision la fréquence fondamentale injectée."""
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    target_freq = 600.0
    sig = np.sin(2 * np.pi * target_freq * t).astype(np.float32)

    res = analyze_frequency_domain(sig, sample_rate=fs, n_fft=2048)
    assert abs(res.dominant_frequency_hz - target_freq) <= (fs / 2048)
    assert 550.0 <= res.spectral_centroid_hz <= 650.0
    assert 0.0 <= res.spectral_entropy <= 1.0
    assert 0.0 <= res.spectral_flatness <= 1.0


def test_stft_and_mel_spectrogram_shapes():
    """Vérifie les dimensions des matrices STFT et Mel-Spectrogramme."""
    fs = 16000
    sig = np.random.randn(fs).astype(np.float32)
    n_fft = 512
    hop_length = 128
    n_mels = 64

    res = analyze_time_frequency(
        sig,
        sample_rate=fs,
        n_fft=n_fft,
        win_length=512,
        hop_length=hop_length,
        n_mels=n_mels
    )

    expected_freq_bins = n_fft // 2 + 1  # 257
    expected_frames = 1 + (fs // hop_length)

    assert res.spectrogram_db.shape[0] == expected_freq_bins
    assert res.mel_spectrogram_db.shape[0] == n_mels
    assert abs(res.spectrogram_db.shape[1] - expected_frames) <= 2


def test_mfcc_extraction_dimensions():
    """Vérifie le calcul des 13 coefficients MFCC et de leurs dérivées."""
    fs = 16000
    sig = np.random.randn(fs).astype(np.float32)

    mfcc, delta, delta2, times = compute_mfcc_features(
        sig,
        sample_rate=fs,
        n_mfcc=13,
        n_fft=512,
        hop_length=128
    )

    assert mfcc.shape[0] == 13
    assert delta.shape[0] == 13
    assert delta2.shape[0] == 13
    assert mfcc.shape[1] == len(times)


def test_feature_vector_completeness():
    """Vérifie l'intégrité et la complétude du vecteur de 91 descripteurs numériques."""
    fs = 16000
    sig = (np.sin(np.linspace(0, 100, fs)) + 0.1 * np.random.randn(fs)).astype(np.float32)

    feat_res = extract_features(sig, sample_rate=fs, n_mfcc=13, include_deltas=True)

    assert len(feat_res.feature_names) == 91
    assert len(feat_res.feature_vector) == 91
    assert feat_res.feature_dataframe.shape == (1, 91)

    # Aucune valeur manquante ou infinie
    assert np.isfinite(feat_res.feature_vector).all()
    for name, val in feat_res.feature_dict.items():
        assert isinstance(val, (float, int, np.floating))
        assert not np.isnan(val)
        assert not np.isinf(val)
