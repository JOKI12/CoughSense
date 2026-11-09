"""
Module de détection et segmentation des épisodes de toux pour CoughSense.

Implémente un détecteur basé sur l'enveloppe énergétique locale (RMS court-terme)
avec hystérésis temporelle et seuillage statistique adaptatif.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass
class CoughSegmentResult:
    """Résultat de la détection et segmentation de la toux."""
    segmented_signal: np.ndarray       # Signal audio découpé sur l'intervalle de toux
    start_sample: int                  # Indice de début dans le signal prétraité
    end_sample: int                    # Indice de fin dans le signal prétraité
    start_time: float                  # Début en secondes
    end_time: float                    # Fin en secondes
    cough_detected: bool               # Vrai si au moins un sursaut d'énergie caractéristique a été validé
    energy_envelope: np.ndarray        # Enveloppe d'énergie calculée
    envelope_times: np.ndarray         # Axe temporel correspondant à l'enveloppe
    threshold: float                   # Seuil d'énergie appliqué pour la détection
    num_bursts: int                    # Nombre de sursauts/quintes détectés


def compute_energy_envelope(
    signal: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 128
) -> np.ndarray:
    """
    Calcule l'enveloppe d'énergie RMS court-terme d'un signal audio.

    RMS(m) = sqrt( (1 / N) * sum_{n=0}^{N-1} x[m * H + n]^2 )

    Args:
        signal: Signal 1D.
        frame_length: Longueur de la fenêtre d'analyse (N).
        hop_length: Pas d'avancement temporel (H).

    Returns:
        np.ndarray: Vecteur 1D de l'enveloppe énergétique RMS.
    """
    signal = np.asarray(signal, dtype=np.float32)
    sig_len = len(signal)
    if sig_len < frame_length:
        return np.array([np.sqrt(np.mean(signal**2))], dtype=np.float32)

    n_frames = 1 + (sig_len - frame_length) // hop_length
    rms_values = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = signal[start:end]
        rms_values[i] = np.sqrt(np.mean(frame**2) + 1e-12)

    # Lissage léger de l'enveloppe pour éliminer les micro-fluctuations
    smoothed_rms = uniform_filter1d(rms_values, size=5)
    return np.asarray(smoothed_rms, dtype=np.float32)


def segment_cough(
    signal: np.ndarray,
    sample_rate: int = 16000,
    frame_length: int = 512,
    hop_length: int = 128,
    threshold_multiplier: float = 1.25,
    min_event_duration_sec: float = 0.10,
    max_merge_gap_sec: float = 0.15,
    pad_duration_sec: float = 0.05
) -> CoughSegmentResult:
    """
    Isole la région temporelle correspondant au bruit de toux.

    Principe algorithmique :
    1. Calcul de l'enveloppe RMS court-terme
    2. Calcul d'un seuil adaptatif basé sur la distribution de l'énergie :
       Seuil = mediane(RMS) + multiplier * MAD(RMS)
    3. Détection des intervalles d'activation continue
    4. Fusion des rafales de toux rapprochées (< max_merge_gap_sec)
    5. Filtrage des événements trop brefs (< min_event_duration_sec)
    6. Ajout d'une marge de garde (pad_duration_sec) aux extrémités
    7. Extraction du segment principal

    Args:
        signal: Signal 1D prétraité.
        sample_rate: Fréquence d'échantillonnage en Hz.
        frame_length: Taille de fenêtre RMS en échantillons.
        hop_length: Pas d'avancement RMS en échantillons.
        threshold_multiplier: Multiplicateur statistique pour le seuil adaptatif.
        min_event_duration_sec: Durée minimale requise pour un événement de toux.
        max_merge_gap_sec: Écart maximal entre deux sursauts pour les fusionner en un même épisode.
        pad_duration_sec: Marge de sécurité temporelle ajoutée au début et à la fin.

    Returns:
        CoughSegmentResult: Conteneur du segment extrait et des paramètres de détection.
    """
    total_samples = len(signal)
    total_duration = total_samples / float(sample_rate)

    if total_samples <= frame_length:
        return CoughSegmentResult(
            segmented_signal=signal.copy(),
            start_sample=0,
            end_sample=total_samples,
            start_time=0.0,
            end_time=total_duration,
            cough_detected=True,
            energy_envelope=np.array([float(np.sqrt(np.mean(signal**2)))], dtype=np.float32),
            envelope_times=np.array([0.0], dtype=np.float32),
            threshold=0.0,
            num_bursts=1
        )

    # 1. Calcul de l'enveloppe
    envelope = compute_energy_envelope(signal, frame_length=frame_length, hop_length=hop_length)
    n_frames = len(envelope)
    envelope_times = (np.arange(n_frames) * hop_length + frame_length // 2) / float(sample_rate)

    # 2. Seuil statistique adaptatif (Médiane + écart absolu médian pour robustesse aux bruits impulsifs)
    median_env = float(np.median(envelope))
    mad_env = float(np.median(np.abs(envelope - median_env)))
    threshold = float(median_env + threshold_multiplier * max(mad_env, 0.01))

    # 3. Masque d'activation binaire
    is_active = envelope > threshold

    # 4. Identification des blocs contigus
    events: List[Tuple[int, int]] = []
    in_event = False
    start_frame = 0

    for idx, active in enumerate(is_active):
        if active and not in_event:
            in_event = True
            start_frame = idx
        elif not active and in_event:
            in_event = False
            events.append((start_frame, idx))

    if in_event:
        events.append((start_frame, n_frames - 1))

    # 5. Fusion des événements rapprochés
    merge_frames_gap = int((max_merge_gap_sec * sample_rate) / hop_length)
    min_event_frames = int((min_event_duration_sec * sample_rate) / hop_length)

    merged_events: List[Tuple[int, int]] = []
    for evt in events:
        if not merged_events:
            merged_events.append(evt)
        else:
            prev_start, prev_end = merged_events[-1]
            curr_start, curr_end = evt
            if curr_start - prev_end <= merge_frames_gap:
                merged_events[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged_events.append(evt)

    # Filtrer par durée minimale
    valid_events = [
        (s, e) for (s, e) in merged_events
        if (e - s) >= max(1, min_event_frames)
    ]

    pad_samples = int(pad_duration_sec * sample_rate)

    # Si aucun événement net n'a dépassé le seuil, on conserve l'ensemble du signal
    if not valid_events:
        return CoughSegmentResult(
            segmented_signal=signal.copy(),
            start_sample=0,
            end_sample=total_samples,
            start_time=0.0,
            end_time=total_duration,
            cough_detected=False,
            energy_envelope=envelope,
            envelope_times=envelope_times,
            threshold=threshold,
            num_bursts=0
        )

    # Sélection de la plage englobant les événements détectés
    global_start_frame = valid_events[0][0]
    global_end_frame = valid_events[-1][1]

    start_sample = max(0, int(global_start_frame * hop_length) - pad_samples)
    end_sample = min(total_samples, int(global_end_frame * hop_length + frame_length) + pad_samples)

    segmented_signal = signal[start_sample:end_sample].copy()

    return CoughSegmentResult(
        segmented_signal=segmented_signal,
        start_sample=start_sample,
        end_sample=end_sample,
        start_time=start_sample / float(sample_rate),
        end_time=end_sample / float(sample_rate),
        cough_detected=True,
        energy_envelope=envelope,
        envelope_times=envelope_times,
        threshold=threshold,
        num_bursts=len(valid_events)
    )
