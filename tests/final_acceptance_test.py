"""
Script de validation finale et d'acceptation rigoureuse pour CoughSense.
Exécute les 7 tests d'acceptation de la Section 39 du cahier des charges.
"""

import io
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_loader import AudioValidationError, load_and_standardize_audio
from src.cough_detection import segment_cough
from src.features import extract_features
from src.inference import predict_audio_file
from src.preprocessing import preprocess_signal
from src.signal_analysis import analyze_signal


def run_acceptance_tests():
    print("==================================================================")
    print(" CoughSense - Tests d'Acceptation Finaux (Section 39)")
    print("==================================================================")

    # TEST 1 & 2 : Chargement d'un fichier audio réel de toux
    print("\n--- TEST 1 & 2 : Chargement d'un enregistrement réel ---")
    real_sample_1 = Path("data/raw/coughvid/cough.wav")
    assert real_sample_1.exists(), f"Fichier {real_sample_1} introuvable."
    print(f"Chargement de : {real_sample_1}")

    audio_1 = load_and_standardize_audio(real_sample_1, target_sample_rate=16000)
    print(f"  [OK] Fréquence d'origine: {audio_1.original_sample_rate} Hz -> Standardisée: {audio_1.sample_rate} Hz")
    print(f"  [OK] Durée: {audio_1.duration:.2f} s | Canaux d'origine: {audio_1.original_channels}")

    # TEST 3 : Vérification de la chaîne de traitement complète
    print("\n--- TEST 3 : Prétraitement, FFT, STFT, Spectrogramme, Mel, MFCC ---")
    prep_1 = preprocess_signal(audio_1.signal, sample_rate=16000)
    print(f"  [OK] Prétraitement : Amplitude crête normalisée = {np.max(np.abs(prep_1.processed_signal)):.4f}")
    print(f"  [OK] Débruitage spectral : SNR estimé = {prep_1.snr_estimated_db:.2f} dB")

    seg_1 = segment_cough(prep_1.processed_signal, sample_rate=16000)
    print(f"  [OK] Segmentation de toux : Détectée={seg_1.cough_detected}, Sursauts={seg_1.num_bursts}, Intervalle=[{seg_1.start_time:.2f}s - {seg_1.end_time:.2f}s]")

    analysis_1 = analyze_signal(seg_1.segmented_signal, sample_rate=16000)
    print(f"  [OK] FFT réelle : Fréquence dominante = {analysis_1.frequency_domain.dominant_frequency_hz:.1f} Hz")
    print(f"  [OK] Descripteurs spectraux : Centroïde = {analysis_1.frequency_domain.spectral_centroid_hz:.1f} Hz, Rolloff = {analysis_1.frequency_domain.spectral_rolloff_hz:.1f} Hz")
    print(f"  [OK] STFT Spectrogramme : Matrice {analysis_1.time_frequency.spectrogram_db.shape}")
    print(f"  [OK] Mel-Spectrogramme : Matrice {analysis_1.time_frequency.mel_spectrogram_db.shape}")

    feat_1 = extract_features(seg_1.segmented_signal, sample_rate=16000)
    print(f"  [OK] MFCCs & Caractéristiques : {len(feat_1.feature_names)} descripteurs extraits (Matrice MFCC: {feat_1.mfcc_matrix.shape})")

    # TEST 4 : Inférence ML sur le premier échantillon
    print("\n--- TEST 4 : Inférence ML réelle sur Échantillon 1 ---")
    res_1 = predict_audio_file(real_sample_1, model_name="best")
    print(f"  [OK] Modèle chargé : {res_1.model_name}")
    print(f"  [OK] Prédiction générée : {res_1.predicted_label}")
    print(f"  [OK] Probabilités calculées : {res_1.probabilities}")
    print(f"  [OK] Score de confiance : {res_1.confidence_score * 100:.2f} %")
    print(f"  [OK] Temps d'inférence DSP+ML : {res_1.inference_time_ms:.2f} ms")
    assert np.isclose(sum(res_1.probabilities.values()), 1.0, atol=1e-2)

    # TEST 5 : Test sur un second enregistrement complètement différent
    print("\n--- TEST 5 : Inférence indépendante sur Échantillon 2 ---")
    import glob
    coswara_samples = glob.glob("data/raw/coswara/**/cough-shallow.wav", recursive=True)
    assert len(coswara_samples) > 0, "Aucun échantillon Coswara disponible."
    real_sample_2 = Path(coswara_samples[0])
    print(f"Chargement du second fichier : {real_sample_2}")

    res_2 = predict_audio_file(real_sample_2, model_name="best")
    print(f"  [OK] Prédiction Échantillon 2 : {res_2.predicted_label}")
    print(f"  [OK] Probabilités Échantillon 2 : {res_2.probabilities}")
    print(f"  [OK] Centroïde Échantillon 2 : {res_2.signal_analysis.frequency_domain.spectral_centroid_hz:.1f} Hz (vs {res_1.signal_analysis.frequency_domain.spectral_centroid_hz:.1f} Hz)")
    # Vérification d'indépendance stricte
    assert res_1.audio_data.source_name != res_2.audio_data.source_name

    # TEST 6 : Rejet des fichiers invalides / corrompus
    print("\n--- TEST 6 : Gestion des erreurs et fichiers invalides ---")
    buf_silent = io.BytesIO()
    sf.write(buf_silent, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    buf_silent.seek(0)
    try:
        predict_audio_file(buf_silent)
        raise AssertionError("Erreur : le signal silencieux aurait dû être rejeté !")
    except AudioValidationError as e:
        print(f"  [OK] Rejet attendu du fichier silencieux avec message explicite : {e}")

    buf_short = io.BytesIO()
    sf.write(buf_short, np.ones(500, dtype=np.float32), 16000, format="WAV")
    buf_short.seek(0)
    try:
        predict_audio_file(buf_short)
        raise AssertionError("Erreur : le signal trop court aurait dû être rejeté !")
    except AudioValidationError as e:
        print(f"  [OK] Rejet attendu du fichier trop court avec message explicite : {e}")

    # TEST 7 : Exécution complète de Pytest
    print("\n--- TEST 7 : Suite Pytest automatisée ---")
    import pytest
    ret_code = pytest.main(["-v", "tests/"])
    assert ret_code == 0, "La suite de tests pytest a échoué !"
    print("  [OK] Tous les tests unitaires et d'intégration pytest ont réussi !")

    print("\n==================================================================")
    print(" TOUS LES TESTS D'ACCEPTATION (1 À 7) ONT RÉUSSI AVEC SUCCÈS !")
    print("==================================================================")


if __name__ == "__main__":
    run_acceptance_tests()
