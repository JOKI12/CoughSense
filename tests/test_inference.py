"""
Tests d'intégration pour le moteur d'inférence et la chaîne de prédiction.
"""

import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.audio_loader import AudioValidationError
from src.inference import (
    CoughInferenceEngine,
    InferenceResult,
    ModelNotTrainedError,
    predict_audio_file,
)


def create_synthetic_wav_buffer(duration: float = 1.0, sample_rate: int = 16000) -> io.BytesIO:
    """Crée un buffer WAV valide en mémoire pour les tests d'inférence."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    # Burst de toux synthétique
    sig = 0.5 * np.sin(2 * np.pi * 350 * t) + 0.1 * np.random.randn(len(t))
    buf = io.BytesIO()
    sf.write(buf, sig.astype(np.float32), sample_rate, format="WAV")
    buf.seek(0)
    return buf


def test_inference_engine_loads_model():
    """Vérifie que le moteur d'inférence charge correctement les modèles entraînés."""
    engine = CoughInferenceEngine()
    assert engine.is_model_available("best") is True
    assert engine.is_model_available("SVM") is True
    assert engine.is_model_available("Random_Forest") is True
    assert engine.is_model_available("KNN") is True


def test_end_to_end_inference():
    """Vérifie l'exécution complète du pipeline d'inférence sur un enregistrement en mémoire."""
    buf = create_synthetic_wav_buffer(duration=1.2, sample_rate=16000)
    result = predict_audio_file(buf, model_name="best")

    assert isinstance(result, InferenceResult)
    assert result.predicted_class_id in [0, 1]
    assert result.predicted_label in [
        "Pattern sain / normal",
        "Pattern évocateur d'anomalie respiratoire"
    ]
    assert "Sain / Normal" in result.probabilities
    assert "Anomalie Respiratoire" in result.probabilities
    prob_sum = result.probabilities["Sain / Normal"] + result.probabilities["Anomalie Respiratoire"]
    assert np.isclose(prob_sum, 1.0, atol=1e-2)
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.features_df.shape[1] == 91
    assert result.inference_time_ms > 0.0


def test_inference_on_real_dataset_audio():
    """Vérifie l'inférence sur un enregistrement réel issu du dataset."""
    real_sample = Path("data/raw/coughvid/cough.wav")
    if real_sample.exists():
        result = predict_audio_file(real_sample, model_name="SVM")
        assert result.predicted_class_id in [0, 1]
        assert result.signal_analysis.frequency_domain.dominant_frequency_hz > 0.0
        assert result.cough_segment.num_bursts >= 1


def test_inference_rejects_silent_audio():
    """Vérifie que l'inférence lève une erreur explicite pour un enregistrement silencieux."""
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    buf.seek(0)

    with pytest.raises(AudioValidationError):
        predict_audio_file(buf, model_name="best")


def test_inference_invalid_model_name():
    """Vérifie que le moteur rejette un modèle inexistant."""
    buf = create_synthetic_wav_buffer()
    with pytest.raises(ModelNotTrainedError):
        predict_audio_file(buf, model_name="modele_inexistant_xyz")
