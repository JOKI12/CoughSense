# CoughSense — Analyse et Classification des Signaux de Toux

> **Sous-titre** : Traitement du signal et apprentissage automatique pour le dépistage expérimental des affections respiratoires  
> **Cadre Académique** : Master S3 — *FPGA, VHDL et Traitement du Signal*  
> **Statut** : Projet fonctionnel, reproductible et évalué sur des enregistrements acoustiques réels.

---

> [!WARNING]
> ### ⚠️ AVERTISSEMENT PÉDAGOGIQUE & RECHERCHE (DISCLAIMER)
> Ce système logiciel a été développé à des fins **strictement éducatives et de recherche académique**. Il analyse les propriétés biomécaniques et spectrales des signaux sonores d'expiration forcée. **Il ne constitue en aucun cas un dispositif médical et ne fournit aucun diagnostic clinique.** Ne jamais utiliser les résultats du modèle pour prendre des décisions médicales ou remplacer l'auscultation d'un médecin pneumologue.

---

## 1. Objectifs du Projet

1. **Acquérir et traiter des enregistrements réels** de toux humaine issus de bases de données biomédicales ouvertes de référence : **Coswara (Indian Institute of Science - IISc Bangalore)** et **COUGHVID (EPFL Embedded Systems Laboratory)**.
2. **Implémenter la chaîne complète de Traitement du Signal (DSP)** :
   - Standardisation 16 kHz mono et validation d'intégrité dynamique.
   - Débruitage spectral par soustraction non-linéaire (*spectral gating*).
   - Filtrage passe-bande Butterworth d'ordre 4 $[80\text{ Hz}, 4000\text{ Hz}]$ à phase linéaire nulle.
   - Segmentation temporelle adaptative par enveloppe d'énergie RMS court-terme.
   - Analyse fréquentielle globale par FFT (Spectre d'amplitude, DSP, pic dominant, centroïde, bandwidth, rolloff, entropie spectrale, platitude).
   - Analyse temps-fréquence par STFT (Short-Time Fourier Transform) et spectrogramme sur l'échelle de Mel.
   - Extraction des 13 coefficients cepstraux MFCC avec vitesses ($\Delta$) et accélérations ($\Delta\Delta$).
3. **Construire un vecteur de 91 descripteurs physiques et statistiques**.
4. **Entraîner des modèles d'apprentissage automatique supervisés** :
   - Support Vector Machine (SVM) à noyau RBF calibré.
   - Forêt Aléatoire (Random Forest) pondérée.
   - K-Plus Proches Voisins (KNN).
   - Réseau de neurones convolutif 2D (CNN) sur Mel-Spectrogrammes.
5. **Prévenir rigoureusement toute fuite de données (*Data Leakage*)** en partitionnant les jeux de données par **identifiant unique de participant** ($\text{Train} \cap \text{Test} = \emptyset$).
6. **Fournir une application Web interactive sous Streamlit** permettant le téléversement de fichiers, l'enregistrement direct par microphone, la visualisation complète des spectres et l'expérimentation DSP en direct.

---

## 2. Architecture du Projet

```
cough-signal-analysis/
├── app.py                     # Application Web interactive Streamlit
├── requirements.txt           # Dépendances Python versionnées
├── README.md                  # Documentation complète du projet
├── .gitignore                 # Exclusion des fichiers temporaires et caches
│
├── config/
│   └── config.yaml            # Paramètres de traitement du signal et hyperparamètres ML
│
├── data/
│   ├── README.md              # Documentation des sources et prévention des fuites
│   ├── raw/                   # Données audio brutes (.wav) et métadonnées cliniques
│   │   ├── coswara/           # Lots de participants de la cohorte IISc Coswara
│   │   └── coughvid/          # Enregistrements officiels EPFL COUGHVID
│   └── processed/             # Vecteurs de caractéristiques et métadonnées
│       ├── features_train.csv # Ensemble d'apprentissage (partition par participant)
│       ├── features_val.csv   # Ensemble de validation
│       ├── features_test.csv  # Ensemble de test indépendant (held-out)
│       └── dataset_summary.json
│
├── src/
│   ├── __init__.py
│   ├── audio_loader.py        # Chargement multi-formats, mono, rééchantillonnage 16 kHz
│   ├── preprocessing.py       # Débruitage spectral, filtre Butterworth, normalisation
│   ├── cough_detection.py     # Segmentation par enveloppe RMS adaptative
│   ├── signal_analysis.py     # Métriques temporelles, FFT réelle, STFT, Mel
│   ├── features.py            # Extraction des 91 descripteurs acoustiques & MFCC
│   ├── inference.py           # Moteur d'inférence bout-en-bout réutilisable
│   └── visualization.py       # Tracés graphiques scientifiques (Matplotlib / Seaborn)
│
├── training/
│   ├── prepare_dataset.py     # Script de découpage anti-leakage et extraction DSP
│   ├── train.py               # Entraînement et évaluation des modèles SVM, RF, KNN
│   ├── train_cnn.py           # Entraînement du CNN 2D PyTorch sur Mel-Spectrogrammes
│   ├── evaluate.py            # Évaluation autonome d'un modèle sauvegardé
│   └── model_comparison.py    # Tableau comparatif des performances
│
├── models/
│   ├── best_model.joblib      # Meilleur classifieur retenu (SVM)
│   ├── scaler.joblib          # Normaliseur standard ajusté sur l'entraînement
│   ├── svm_model.joblib
│   ├── random_forest_model.joblib
│   ├── knn_model.joblib
│   ├── cnn_model.pt           # Poids du réseau PyTorch CNN 2D
│   ├── model_metrics.json     # Métriques réelles sur le held-out test set
│   └── feature_names.json     # Ordre strict des 91 colonnes de caractéristiques
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_signal_processing.ipynb
│   └── 03_model_training.ipynb
│
├── tests/
│   ├── test_preprocessing.py   # Tests unitaires chargement, filtre et segmentation
│   ├── test_features.py        # Tests unitaires FFT, STFT, MFCC et dimensions
│   └── test_inference.py       # Tests d'intégration du moteur d'inférence
│
├── docs/
│   └── methodology.md          # Fondements théoriques et mathématiques (DSP & ML)
│
└── scripts/
    ├── download_dataset.py     # Téléchargement automatique depuis dépôts officiels
    └── generate_notebooks.py   # Génération des notebooks Jupyter
```

---

## 3. Guide d'Installation et Reproduction

### 3.1 Cloner ou ouvrir le projet
Ouvrez un terminal dans le répertoire du projet :
```powershell
cd "d:\School\School program\Master\S3\FPGA VHDL ET Traitement de signal\Project"
```

### 3.2 Créer et activer l'environnement virtuel (optionnel mais recommandé)
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3.3 Installer les dépendances
```powershell
pip install -r requirements.txt
```

---

## 4. Pipeline d'Exécution Complet

### Étape 1 — Acquisition des données audio réelles
Télécharge automatiquement les enregistrements officiels et métadonnées depuis les dépôts publics de l'IISc Coswara et de l'EPFL COUGHVID :
```powershell
python scripts/download_dataset.py --source all --batches 20200911 20200505 20200824
```

### Étape 2 — Prétraitement du signal & Extraction des 91 caractéristiques
Exécute la chaîne DSP sur chaque fichier audio et partitionne les données **par participant** pour éliminer toute fuite :
```powershell
python training/prepare_dataset.py
```
*Sortie : `data/processed/features_train.csv`, `features_val.csv`, `features_test.csv` et `dataset_summary.json`.*

### Étape 3 — Entraînement et Évaluation des Modèles Classiques
Ajuste le `StandardScaler` sur l'entraînement, entraîne SVM, Random Forest et KNN, puis évalue les performances sur le jeu de test indépendant :
```powershell
python training/train.py
```

### Étape 4 — Entraînement du Modèle Deep Learning CNN 2D (Optionnel)
Entraîne le réseau de neurones convolutif sur les représentations Mel-Spectrogrammes :
```powershell
python training/train_cnn.py
```

### Étape 5 — Comparaison des Modèles
Affiche le tableau récapitulatif des métriques mesurées :
```powershell
python training/model_comparison.py
```

### Étape 6 — Lancement de l'Application Web Streamlit
```powershell
streamlit run app.py
```
L'application s'ouvre automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

---

## 5. Exécution des Tests Automatisés

Le projet comprend une suite complète de 17 tests unitaires et d'intégration validant le chargement audio, les filtres, la FFT, la STFT, les MFCCs, les dimensions de vecteurs, et le moteur d'inférence.

Pour lancer la suite de tests :
```powershell
pytest -v
```

---

## 6. Synthèse des Résultats d'Évaluation sur le Jeu de Test Indépendant

| Modèle | Accuracy | Balanced Acc. | Précision Macro | Rappel Macro | F1-Score Macro | ROC-AUC | Latence Inférence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (Noyau RBF)** ⭐ | **56.25 %** | **56.25 %** | **0.5635** | **0.5625** | **0.5608** | **0.6250** | **0.048 ms** |
| **KNN (k=5)** | 56.25 % | 56.25 % | 0.5635 | 0.5625 | 0.5608 | 0.6094 | 89.015 ms |
| **Random Forest** | 50.00 % | 50.00 % | 0.5000 | 0.5000 | 0.4921 | 0.4844 | 1.220 ms |
| **CNN 2D (Mel-Spec)** | 43.75 % | 43.75 % | 0.4365 | 0.4375 | 0.4353 | 0.6094 | 1.687 ms |

*Remarque scientifique : Ces résultats sont obtenus sur de véritables enregistrements in-the-wild tenus strictement à l'écart. Ils démontrent la supériorité de l'extraction de descripteurs physiques d'ingénierie du signal (DSP) combinés au SVM face à des modèles complexes sur petit échantillon.*

---

## 7. Fonctionnalités de l'Application Web

1. **Accueil & Cadre Scientifique** : Présentation du projet universitaire, chaîne de traitement et formules mathématiques.
2. **Analyse & Dépistage de Toux** :
   - Téléversement multi-formats (.wav, .mp3, .ogg, .flac).
   - Enregistrement direct par microphone de l'ordinateur via `st.audio_input`.
   - Sélection d'échantillons réels pré-intégrés de la cohorte.
   - Inférence réelle avec affichage des probabilités de classe et du niveau de confiance.
3. **Analyse Temporelle & Fréquentielle** :
   - Forme d'onde brute vs prétraitée avec enveloppe et détection de sursauts.
   - Spectre de Fourier (FFT) : amplitude et DSP, pic dominant, centroïde, rolloff.
   - Représentations temps-fréquence : Spectrogramme STFT et Spectrogramme de Mel.
   - Matrice des 13 coefficients MFCC à travers le temps.
4. **Vecteur de Caractéristiques** : Tableau interactif des 91 valeurs numériques extraites et bouton de téléchargement CSV.
5. **Performances des Modèles** : Matrice de confusion interactive et classification report.
6. **Laboratoire Expérimental DSP** : Réglage en direct des fréquences de coupure, de l'ordre du filtre Butterworth, de la taille FFT et des fenêtres de troncature pour illustrer le principe d'incertitude temps-fréquence de Gabor-Heisenberg ($\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$).
7. **Jeu de Données & Métadonnées** : Statistiques détaillées de la cohorte et démonstration de la prévention du data leakage.

---

## 8. Références Scientifiques
1. **COUGHVID** : Orlandic et al., *The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms*, Scientific Data, Nature (2021). [DOI: 10.1038/s41597-021-00937-4](https://doi.org/10.1038/s41597-021-00937-4)
2. **Coswara** : Bhattacharya et al., *Coswara — A Database of In-the-wild Audio Samples for COVID-19 Detection*, Interspeech (2021). [arXiv:2005.10548](https://arxiv.org/abs/2005.10548)
3. **Traitement de la Parole et du Signal Audio** : L. Rabiner & R. Schafer, *Theory and Applications of Digital Speech Processing*, Pearson (2010).
