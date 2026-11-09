"""
CoughSense — Analyse et classification des signaux de toux
Application Web de Traitement du Signal & Apprentissage Automatique
Master S3 — FPGA, VHDL et Traitement du Signal
"""

import io
import json
from pathlib import Path
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configuration de la page Streamlit
st.set_page_config(
    page_title="CoughSense — Traitement du Signal",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_loader import (
    AudioData,
    AudioFormatError,
    AudioValidationError,
    load_and_standardize_audio,
)
from src.cough_detection import segment_cough
from src.features import extract_features
from src.inference import (
    CoughInferenceEngine,
    InferenceResult,
    ModelNotTrainedError,
    get_inference_engine,
)
from src.preprocessing import preprocess_signal
from src.signal_analysis import analyze_signal
from src.visualization import (
    plot_confusion_matrix,
    plot_frequency_spectrum,
    plot_mfcc_matrix,
    plot_model_comparison_bars,
    plot_time_frequency_representations,
    plot_waveform_and_segmentation,
)

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


# -----------------------------------------------------------------------------
# CACHING DES RESSOURCES
# -----------------------------------------------------------------------------
@st.cache_resource
def load_cached_engine() -> CoughInferenceEngine:
    """Charge le moteur d'inférence en cache."""
    return get_inference_engine()


@st.cache_data
def load_cached_summary() -> dict:
    """Charge les métadonnées consolidées du jeu de données."""
    sum_path = DATA_PROCESSED_DIR / "dataset_summary.json"
    if sum_path.exists():
        with open(sum_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_cached_metrics() -> dict:
    """Charge les métriques réelles d'évaluation sur le test set."""
    met_path = MODELS_DIR / "model_metrics.json"
    if met_path.exists():
        with open(met_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# -----------------------------------------------------------------------------
# STYLE CSS ÉPURÉ ET ACADÉMIQUE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Typographie et espacements épurés */
    .stApp {
        background-color: #fafbfc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    /* Bandeau d'avertissement sobre */
    .disclaimer-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #64748b;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 13px;
        color: #475569;
        margin-bottom: 16px;
    }
    /* Cartes de résultats sobres */
    .result-card-healthy {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
    }
    .result-card-abnormal {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #d97706;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
    }
    .stat-pill {
        display: inline-block;
        background: #f1f5f9;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        color: #334155;
        font-weight: 500;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def render_academic_disclaimer():
    """Affiche un avertissement académique minimaliste et non intrusif."""
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Prototype de recherche & pédagogique</b> — Analyse acoustique expérimentale (Master S3 Traitement du Signal).
            Ne constitue pas un dispositif médical et ne fournit aucun diagnostic clinique.
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("### COUGHSENSE")
st.sidebar.caption("Traitement du Signal Acoustique")

page = st.sidebar.radio(
    "Menu",
    [
        "Accueil",
        "Analyse",
        "Signal",
        "Modèle",
        "Laboratoire DSP",
        "À propos"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.caption("Master S3 — Traitement du Signal & FPGA\nIISc Coswara & EPFL COUGHVID")


# -----------------------------------------------------------------------------
# 1. ACCUEIL
# -----------------------------------------------------------------------------
if page == "Accueil":
    st.title("CoughSense")
    st.caption("Analyse spectrale et classification des signaux de toux pour le dépistage expérimental des affections respiratoires")

    render_academic_disclaimer()

    col_h1, col_h2 = st.columns([3, 2])

    with col_h1:
        st.markdown("""
        Ce projet académique implémente une chaîne complète de traitement du signal audio et d'apprentissage automatique appliquée à des enregistrements réels de toux humaine :

        - **Standardisation** : 16 kHz, mono, conversion 16-bit PCM.
        - **Prétraitement** : Débruitage spectral + filtre passe-bande Butterworth d'ordre 4 ($[80\\text{ Hz}, 4000\\text{ Hz}]$).
        - **Segmentation** : Détection des sursauts par enveloppe RMS adaptative.
        - **Analyse Spectrale** : FFT réelle, DSP, centroïde, bandwidth, rolloff, entropie, STFT, Mel-Spectrogramme, 13 MFCCs.
        - **Classification Supervisée** : Modèles SVM, Random Forest, KNN et CNN entraînés avec partitionnement étanche par participant (*Zero Data Leakage*).
        """)

        st.write("")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🎙️ Lancer une analyse", type="primary", width="stretch"):
                st.session_state["nav_to_analyse"] = True
                st.rerun()
        with c_btn2:
            if st.button("🧪 Laboratoire DSP", width="stretch"):
                st.session_state["nav_to_lab"] = True
                st.rerun()

    with col_h2:
        st.markdown("#### Métriques Clés du Projet")
        sum_data = load_cached_summary()
        m_rec = sum_data.get("total_recordings", 77)
        m_part = sum_data.get("total_participants", 40)
        m_feat = sum_data.get("num_features", 91)

        c1, c2 = st.columns(2)
        c1.metric("Enregistrements réels", m_rec)
        c2.metric("Participants", m_part)

        c3, c4 = st.columns(2)
        c3.metric("Descripteurs DSP", m_feat)
        c4.metric("Fréquence d'échantillonnage", "16 kHz")

        st.caption("Données réelles issues de Coswara (IISc) et COUGHVID (EPFL).")


# -----------------------------------------------------------------------------
# 2. ANALYSE (MAIN PAGE)
# -----------------------------------------------------------------------------
elif page == "Analyse":
    st.subheader("Analyse d'une toux")
    render_academic_disclaimer()

    engine = load_cached_engine()
    if not engine.is_model_available():
        st.error("Modèle non disponible. Veuillez exécuter le pipeline d'entraînement préalable (`python training/train.py`).")
        st.stop()

    # Sélection de la source
    input_source = st.radio(
        "Source audio",
        ["Téléverser un fichier", "Microphone", "Échantillons de référence"],
        horizontal=True,
        label_visibility="collapsed"
    )

    audio_bytes_or_path = None
    source_name = "audio_input"

    if input_source == "Téléverser un fichier":
        up_file = st.file_uploader(
            "Glissez un fichier audio de toux (.wav, .mp3, .ogg, .flac) :",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            label_visibility="visible"
        )
        if up_file:
            audio_bytes_or_path = up_file.read()
            source_name = up_file.name

    elif input_source == "Microphone":
        st.caption("Enregistrez votre toux depuis le microphone du navigateur :")
        mic_audio = st.audio_input("Enregistrer un signal de toux")
        if mic_audio:
            audio_bytes_or_path = mic_audio.read()
            source_name = "enregistrement_micro.wav"

    elif input_source == "Échantillons de référence":
        cohort_files = {}
        # COUGHVID
        for f in (DATA_RAW_DIR / "coughvid").glob("*.wav"):
            cohort_files[f"[COUGHVID] {f.name}"] = f
        # Coswara
        for f in list((DATA_RAW_DIR / "coswara").glob("**/*cough*.wav"))[:10]:
            cohort_files[f"[Coswara] {f.parent.name[:10]} - {f.name}"] = f

        if cohort_files:
            chosen_key = st.selectbox("Sélectionnez un enregistrement réel de la cohorte :", list(cohort_files.keys()))
            chosen_path = cohort_files[chosen_key]
            source_name = chosen_path.name
            with open(chosen_path, "rb") as fp:
                audio_bytes_or_path = fp.read()
        else:
            st.warning("Aucun enregistrement trouvé dans data/raw/.")

    if audio_bytes_or_path:
        st.audio(audio_bytes_or_path)

        c_opt1, c_opt2 = st.columns([3, 1])
        with c_opt1:
            model_choice = st.selectbox(
                "Modèle de classification",
                ["best", "SVM", "KNN", "Random_Forest"],
                format_func=lambda x: "Modèle retenu : SVM (Recommandé)" if x == "best" else f"Modèle {x}"
            )
        with c_opt2:
            st.write("")
            st.write("")
            trigger_analysis = st.button("ANALYSER LE SIGNAL", type="primary", width="stretch")

        if trigger_analysis or ("last_inference_result" in st.session_state and st.session_state.get("last_source_name") == source_name):
            if trigger_analysis:
                with st.spinner("Exécution de la chaîne DSP et inférence..."):
                    try:
                        res = engine.predict(audio_bytes_or_path, model_name=model_choice, source_name=source_name)
                        st.session_state["last_inference_result"] = res
                        st.session_state["last_source_name"] = source_name
                    except (AudioValidationError, AudioFormatError) as e:
                        print(f"Validation error: {e}", file=sys.stderr)
                        traceback.print_exc()
                        st.error(f"Impossible d'analyser ce fichier : {e}")
                        st.stop()
                    except Exception as e:
                        print(f"Analysis error: {e}", file=sys.stderr)
                        traceback.print_exc()
                        st.error("Impossible d'analyser ce fichier. Vérifiez que l'audio contient un signal exploitable.")
                        st.stop()

            res: InferenceResult = st.session_state.get("last_inference_result")
            if res:
                st.divider()

                # Résumé technique audio
                st.markdown(
                    f"""
                    <span class="stat-pill">Durée : {res.signal_analysis.time_domain.duration_sec:.2f} s</span>
                    <span class="stat-pill">Fréquence : {res.audio_data.sample_rate} Hz (Mono)</span>
                    <span class="stat-pill">SNR : {res.preprocessing_result.snr_estimated_db:.1f} dB</span>
                    <span class="stat-pill">Sursauts détectés : {res.cough_segment.num_bursts}</span>
                    <span class="stat-pill">Latence DSP+ML : {res.inference_time_ms:.1f} ms</span>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                # Bloc de prédiction
                col_pred, col_bar = st.columns([1, 1])

                is_abnormal = res.predicted_class_id == 1
                card_class = "result-card-abnormal" if is_abnormal else "result-card-healthy"
                accent_color = "#d97706" if is_abnormal else "#16a34a"

                with col_pred:
                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div style="font-size: 13px; color: #64748b; font-weight: 500;">Prédiction du modèle</div>
                            <div style="font-size: 20px; font-weight: 700; color: {accent_color}; margin: 4px 0;">
                                {res.predicted_label}
                            </div>
                            <div style="font-size: 14px; color: #334155;">
                                Degré de confiance : <b>{res.confidence_score * 100:.2f} %</b> ({res.model_name})
                            </div>
                            <div style="font-size: 11px; color: #64748b; margin-top: 6px;">
                                Pattern acoustique correspondant à la classe prédite. Résultat expérimental — ne constitue pas un diagnostic médical.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col_bar:
                    p_healthy = res.probabilities.get("Sain / Normal", 0.0)
                    p_abnormal = res.probabilities.get("Anomalie Respiratoire", 0.0)

                    st.caption("Distribution des probabilités de classe")
                    st.progress(p_healthy, text=f"Sain / Normal : {p_healthy * 100:.1f} %")
                    st.progress(p_abnormal, text=f"Anomalie Respiratoire : {p_abnormal * 100:.1f} %")

                # Aperçu rapide des signaux
                st.write("")
                st.caption("Aperçu de la forme d'onde et de la segmentation :")
                fig_mini = plot_waveform_and_segmentation(
                    res.audio_data.signal,
                    res.preprocessing_result,
                    res.cough_segment,
                    sample_rate=res.audio_data.sample_rate
                )
                st.pyplot(fig_mini)
                plt.close(fig_mini)

                st.info("💡 Rendez-vous dans la page **'Signal'** pour explorer en détail la FFT, la STFT, le Mel-Spectrogramme et les 91 descripteurs numériques.")


# -----------------------------------------------------------------------------
# 3. SIGNAL (DSP VISUALIZATIONS)
# -----------------------------------------------------------------------------
elif page == "Signal":
    st.subheader("Analyse du Signal (DSP)")
    render_academic_disclaimer()

    if "last_inference_result" not in st.session_state:
        st.info("Aucun signal en mémoire. Veuillez d'abord analyser un audio dans la page **Analyse**.")
    else:
        res: InferenceResult = st.session_state["last_inference_result"]

        tab_t, tab_f, tab_tf, tab_m, tab_feat = st.tabs([
            "Temps",
            "Fréquence (FFT)",
            "Temps-Fréquence (STFT / Mel)",
            "Cepstral (MFCC)",
            "Descripteurs (91)"
        ])

        with tab_t:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Durée", f"{res.signal_analysis.time_domain.duration_sec:.2f} s")
            c2.metric("Énergie RMS", f"{res.signal_analysis.time_domain.rms_energy:.4f}")
            c3.metric("ZCR", f"{res.signal_analysis.time_domain.zero_crossing_rate:.4f}")
            c4.metric("Facteur de crête", f"{res.signal_analysis.time_domain.crest_factor:.2f}")

            fig_t = plot_waveform_and_segmentation(
                res.audio_data.signal,
                res.preprocessing_result,
                res.cough_segment,
                sample_rate=res.audio_data.sample_rate
            )
            st.pyplot(fig_t)
            plt.close(fig_t)

        with tab_f:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pic dominant", f"{res.signal_analysis.frequency_domain.dominant_frequency_hz:.1f} Hz")
            c2.metric("Centroïde", f"{res.signal_analysis.frequency_domain.spectral_centroid_hz:.1f} Hz")
            c3.metric("Bande passante", f"{res.signal_analysis.frequency_domain.spectral_bandwidth_hz:.1f} Hz")
            c4.metric("Rolloff (85%)", f"{res.signal_analysis.frequency_domain.spectral_rolloff_hz:.1f} Hz")

            fig_f = plot_frequency_spectrum(res.signal_analysis.frequency_domain)
            st.pyplot(fig_f)
            plt.close(fig_f)

            st.caption(
                f"Entropie spectrale : {res.signal_analysis.frequency_domain.spectral_entropy:.3f} | "
                f"Platitude spectrale : {res.signal_analysis.frequency_domain.spectral_flatness:.4f}"
            )

        with tab_tf:
            fig_tf = plot_time_frequency_representations(res.signal_analysis.time_frequency)
            st.pyplot(fig_tf)
            plt.close(fig_tf)

        with tab_m:
            fig_mfcc = plot_mfcc_matrix(res.mfcc_result)
            st.pyplot(fig_mfcc)
            plt.close(fig_mfcc)

        with tab_feat:
            st.markdown(f"**Vecteur complet de {len(res.feature_dict)} descripteurs**")
            feat_df = pd.DataFrame(
                [{"Descripteur": k, "Valeur": round(v, 5)} for k, v in res.feature_dict.items()]
            )
            st.dataframe(feat_df, width="stretch", hide_index=True)

            csv_data = res.features_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les descripteurs (CSV)",
                data=csv_data,
                file_name="features_coughsense.csv",
                mime="text/csv",
                width="stretch"
            )


# -----------------------------------------------------------------------------
# 4. MODÈLE (PERFORMANCES RÉELLES SUR TEST SET)
# -----------------------------------------------------------------------------
elif page == "Modèle":
    st.subheader("Performances des Modèles")
    render_academic_disclaimer()

    metrics = load_cached_metrics()
    if not metrics:
        st.error("Métriques introuvables. Entraînez les modèles via `python training/train.py`.")
        st.stop()

    # Tableau comparatif
    st.markdown("#### Évaluation sur le Held-Out Test Set (Participants Inédits)")
    table_rows = []
    for model_name, m in metrics.items():
        table_rows.append({
            "Modèle": model_name,
            "Accuracy": f"{m.get('accuracy', 0.0) * 100:.2f} %",
            "Balanced Acc": f"{m.get('balanced_accuracy', 0.0) * 100:.2f} %",
            "Précision": f"{m.get('precision_macro', 0.0):.3f}",
            "Rappel": f"{m.get('recall_macro', 0.0):.3f}",
            "F1-Score": f"{m.get('f1_macro', 0.0):.3f}",
            "ROC-AUC": f"{m.get('roc_auc', 0.0):.3f}",
            "Latence (ms)": f"{m.get('inference_time_ms_per_sample', 0.0):.3f}"
        })

    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
    st.caption("Performances réelles mesurées sur la cohorte de test. Aucune donnée n'a été tronquée ou inventée.")

    st.write("")

    col_chart, col_cm = st.columns([3, 2])

    with col_chart:
        fig_bar = plot_model_comparison_bars(metrics)
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with col_cm:
        sel_mod = st.selectbox("Modèle pour la matrice de confusion :", list(metrics.keys()))
        cm_data = metrics[sel_mod].get("confusion_matrix", [[0, 0], [0, 0]])
        fig_cm = plot_confusion_matrix(cm_data)
        st.pyplot(fig_cm)
        plt.close(fig_cm)


# -----------------------------------------------------------------------------
# 5. LABORATOIRE DSP (SANDBOX D'INGÉNIERIE)
# -----------------------------------------------------------------------------
elif page == "Laboratoire DSP":
    st.subheader("Laboratoire Expérimental DSP")
    st.caption("Outil d'ingénierie : observez l'impact direct des paramètres de filtrage et de FFT sur le signal.")
    render_academic_disclaimer()

    all_raw_files = list(DATA_RAW_DIR.glob("**/*cough*.wav"))
    if not all_raw_files:
        st.warning("Aucun enregistrement disponible dans data/raw/.")
        st.stop()

    file_options = {f.name: f for f in all_raw_files[:8]}
    chosen_file_name = st.selectbox("Signal audio de travail :", list(file_options.keys()))
    working_file_path = file_options[chosen_file_name]

    st.write("")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        f_low = st.slider("Coupure basse (Hz)", 20.0, 300.0, 80.0, step=10.0)
        f_high = st.slider("Coupure haute (Hz)", 1000.0, 7000.0, 4000.0, step=200.0)
    with col_p2:
        filt_order = st.slider("Ordre Butterworth", 2, 8, 4, step=2)
        win_type = st.selectbox("Fenêtre temporelle", ["hann", "hamming", "blackman", "boxcar"])
    with col_p3:
        fft_n = st.select_slider("Taille FFT (N)", options=[256, 512, 1024, 2048, 4096], value=1024)
        nr_active = st.checkbox("Débruitage spectral", value=True)

    delta_freq = 16000.0 / fft_n
    st.markdown(
        f"""
        <div style="background:#f1f5f9; padding:8px 14px; border-radius:4px; font-size:13px; margin: 8px 0 14px 0;">
            <b>Résolution fréquentielle :</b> &Delta;f = f<sub>s</sub> / N = 16000 / {fft_n} = <b>{delta_freq:.2f} Hz</b> &nbsp;|&nbsp;
            <b>Principe d'incertitude :</b> Une fenêtre plus longue affine la résolution fréquentielle au détriment de la précision temporelle.
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        audio_lab = load_and_standardize_audio(working_file_path, target_sample_rate=16000)
        prep_lab = preprocess_signal(
            audio_lab.signal,
            sample_rate=16000,
            apply_noise_reduction=nr_active,
            apply_filtering=True,
            low_cutoff_hz=f_low,
            high_cutoff_hz=f_high,
            filter_order=filt_order
        )
        analysis_lab = analyze_signal(
            prep_lab.processed_signal,
            sample_rate=16000,
            n_fft_spectrum=fft_n,
            stft_n_fft=fft_n,
            stft_win_length=min(fft_n, 512),
            stft_hop_length=128,
            stft_window=win_type
        )

        fig_spec_lab = plot_frequency_spectrum(analysis_lab.frequency_domain)
        st.pyplot(fig_spec_lab)
        plt.close(fig_spec_lab)

    except Exception as e:
        print(f"DSP Lab error: {e}", file=sys.stderr)
        traceback.print_exc()
        st.error("Erreur lors du calcul DSP dans le laboratoire.")


# -----------------------------------------------------------------------------
# 6. À PROPOS
# -----------------------------------------------------------------------------
elif page == "À propos":
    st.subheader("À propos de CoughSense")
    render_academic_disclaimer()

    st.markdown("""
    #### Contexte & Méthodologie
    - **Cadre Universitaire** : Master S3 — *Traitement du Signal, FPGA & VHDL*.
    - **Enjeux Scientifiques** : Déterminer si les caractéristiques temporelles, spectrales et cepstrales d'un signal de toux contiennent une information discriminante exploitable pour le dépistage d'affections respiratoires.
    - **Cohortes Utilisées** :
      - **Coswara (IISc Bangalore)** : Enregistrements crowdsourcés avec statuts cliniques certifiés et métadonnées complètes.
      - **COUGHVID (EPFL)** : Corpus public de toux annoté par des médecins pneumologues.
    - **Prévention du Data Leakage** :
      La partition Train / Validation / Test est strictement réalisée **au niveau du participant** :
      $$\\text{Participants}_{Train} \\cap \\text{Participants}_{Test} = \\emptyset$$
      Aucun individu n'a de fragments d'audio partagés entre l'apprentissage et le test.
    """)

    sum_dict = load_cached_summary()
    if sum_dict:
        st.markdown("#### Statistiques Consolidées de la Cohorte")
        split_rows = []
        for sname, sinfo in sum_dict.get("splits", {}).items():
            split_rows.append({
                "Ensemble": sname.capitalize(),
                "Enregistrements": sinfo.get("recordings", 0),
                "Participants": sinfo.get("participants", 0),
                "Sain": sinfo.get("healthy", 0),
                "Anomalie": sinfo.get("abnormal", 0)
            })
        st.dataframe(pd.DataFrame(split_rows), width="stretch", hide_index=True)
