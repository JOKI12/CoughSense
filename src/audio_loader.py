"""
Module de chargement et de validation audio pour CoughSense.

Prend en charge les formats .wav, .mp3, .m4a, .flac, .ogg, la conversion en mono,
le rééchantillonnage standardisé à 16 kHz et la validation rigoureuse du signal.
"""

from dataclasses import dataclass
import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf


class AudioValidationError(ValueError):
    """Exception levée lorsque le signal audio ne satisfait pas les critères de validité."""
    pass


class AudioFormatError(ValueError):
    """Exception levée en cas de format de fichier non supporté ou corrompu."""
    pass


@dataclass
class AudioData:
    """Structure encapsulant le signal audio et ses métadonnées techniques."""
    signal: np.ndarray             # Signal audio standardisé (mono, float32, 16 kHz)
    sample_rate: int               # Fréquence d'échantillonnage finale (ex: 16000 Hz)
    duration: float                # Durée finale en secondes
    original_sample_rate: int      # Fréquence d'échantillonnage d'origine
    original_duration: float       # Durée d'origine en secondes
    original_channels: int         # Nombre de canaux d'origine (1=mono, 2=stéréo)
    source_name: str               # Nom ou identifiant du fichier source
    peak_amplitude: float          # Amplitude crête du signal standardisé
    rms_energy: float              # Énergie RMS du signal standardisé


def validate_audio_signal(
    signal: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 0.25,
    max_duration_sec: float = 15.0,
    silence_threshold: float = 1e-4
) -> None:
    """
    Valide l'intégrité scientifique et technique d'un signal audio.

    Args:
        signal: Tableau 1D numpy représentant les échantillons audio.
        sample_rate: Fréquence d'échantillonnage en Hz.
        min_duration_sec: Durée minimale acceptable en secondes.
        max_duration_sec: Durée maximale acceptable en secondes.
        silence_threshold: Seuil d'amplitude crête en-deçà duquel le signal est considéré silencieux.

    Raises:
        AudioValidationError: Si le signal viole l'un des critères de qualité.
    """
    if signal is None or len(signal) == 0:
        raise AudioValidationError("Le signal audio est vide (0 échantillon).")

    if not np.isfinite(signal).all():
        raise AudioValidationError("Le signal contient des valeurs invalides (NaN ou Infini).")

    duration = len(signal) / float(sample_rate)

    if duration < min_duration_sec:
        raise AudioValidationError(
            f"Enregistrement trop court ({duration:.2f} s). Durée minimale requise : {min_duration_sec:.2f} s."
        )

    if duration > max_duration_sec:
        raise AudioValidationError(
            f"Enregistrement trop long ({duration:.2f} s). Durée maximale autorisée : {max_duration_sec:.2f} s."
        )

    peak = float(np.max(np.abs(signal)))
    if peak < silence_threshold:
        raise AudioValidationError(
            f"Signal quasi-silencieux ou inaudible détecté (amplitude crête = {peak:.6f} < {silence_threshold})."
        )


def load_and_standardize_audio(
    source: Union[str, Path, bytes, io.BytesIO],
    target_sample_rate: int = 16000,
    min_duration_sec: float = 0.25,
    max_duration_sec: float = 15.0,
    source_name: Optional[str] = None
) -> AudioData:
    """
    Charge et standardise un signal audio depuis un fichier, un flux binaire ou un buffer en mémoire.

    Étapes de standardisation :
    1. Lecture du flux (SoundFile ou fallback Librosa/Audioread)
    2. Extraction des métadonnées d'origine (fréquence, canaux, durée)
    3. Conversion stéréo -> mono par moyenne arithmétique des voies
    4. Rééchantillonnage sinc / polyphase à la fréquence cible (16 kHz par défaut)
    5. Normalisation du type de données en float32 [-1.0, 1.0]
    6. Validation stricte des critères de qualité

    Args:
        source: Chemin de fichier, bytes ou objet BytesIO (provenant de st.file_uploader ou st.audio_input)
        target_sample_rate: Fréquence d'échantillonnage cible (défaut: 16000 Hz)
        min_duration_sec: Durée minimale requise
        max_duration_sec: Durée maximale autorisée
        source_name: Nom optionnel pour l'affichage

    Returns:
        AudioData: Conteneur complet du signal prêt pour le prétraitement et l'analyse.
    """
    if source_name is None:
        if isinstance(source, (str, Path)):
            source_name = Path(source).name
        else:
            source_name = "audio_input"

    raw_signal = None
    orig_sr = None
    orig_channels = 1

    # 1. Tentative de lecture via SoundFile (rapide et précis pour WAV, FLAC, OGG)
    file_obj = source
    if isinstance(source, bytes):
        file_obj = io.BytesIO(source)

    try:
        raw_signal, orig_sr = sf.read(file_obj, dtype="float32", always_2d=False)
        if raw_signal.ndim == 2:
            orig_channels = raw_signal.shape[1]
            # Conversion en mono par moyenne arithmétique des voies
            raw_signal = np.mean(raw_signal, axis=1)
        else:
            orig_channels = 1
    except Exception:
        # 2. Fallback via Librosa (idéal pour MP3, WebM, M4A, etc.)
        try:
            if isinstance(source, (bytes, io.BytesIO)):
                if isinstance(source, io.BytesIO):
                    source.seek(0)
                    buffer = source
                else:
                    buffer = io.BytesIO(source)
                raw_signal, orig_sr = librosa.load(buffer, sr=None, mono=True)
            else:
                raw_signal, orig_sr = librosa.load(str(source), sr=None, mono=True)
            orig_channels = 1
        except Exception as e:
            raise AudioFormatError(
                f"Impossible de décoder le flux audio '{source_name}'. "
                f"Formats acceptés : WAV, MP3, FLAC, OGG, WebM, M4A. Erreur sous-jacente : {e}"
            )

    if raw_signal is None or orig_sr is None:
        raise AudioFormatError(f"Le décodage de l'audio '{source_name}' a échoué.")

    raw_signal = np.asarray(raw_signal, dtype=np.float32)
    orig_duration = len(raw_signal) / float(orig_sr)

    # 3. Rééchantillonnage haute fidélité si la fréquence diffère de la cible
    if orig_sr != target_sample_rate:
        signal_resampled = librosa.resample(
            raw_signal,
            orig_sr=orig_sr,
            target_sr=target_sample_rate,
            res_type="soxr_hq" if hasattr(librosa, "soxr") else "kaiser_best"
        )
    else:
        signal_resampled = raw_signal

    signal_resampled = np.asarray(signal_resampled, dtype=np.float32)
    final_duration = len(signal_resampled) / float(target_sample_rate)

    # 4. Validation du signal standardisé
    validate_audio_signal(
        signal_resampled,
        sample_rate=target_sample_rate,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec
    )

    peak = float(np.max(np.abs(signal_resampled)))
    rms = float(np.sqrt(np.mean(signal_resampled**2)))

    return AudioData(
        signal=signal_resampled,
        sample_rate=target_sample_rate,
        duration=final_duration,
        original_sample_rate=orig_sr,
        original_duration=orig_duration,
        original_channels=orig_channels,
        source_name=str(source_name),
        peak_amplitude=peak,
        rms_energy=rms
    )
