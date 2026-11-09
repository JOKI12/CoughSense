# Organisation des Données (Data Directory)

Ce répertoire contient les données audio et métadonnées utilisées pour l'entraînement, la validation et l'évaluation du système **CoughSense**.

## Structure

```
data/
├── raw/                      # Fichiers audio bruts (.wav, .ogg) et métadonnées originales
│   ├── coswara/              # Sous-ensemble officiel Coswara (IISc Bangalore)
│   └── coughvid/             # Enregistrements officiels COUGHVID (EPFL)
│
├── processed/                # Fichiers nettoyés, segmentés et vecteurs de caractéristiques
│   ├── features_train.csv    # Vecteurs de caractéristiques de l'ensemble d'entraînement
│   ├── features_val.csv      # Vecteurs de caractéristiques de l'ensemble de validation
│   ├── features_test.csv     # Vecteurs de caractéristiques de l'ensemble de test
│   └── dataset_metadata.json # Métadonnées consolidées et statistiques de split
```

## Sources Officielles

1. **COUGHVID Dataset (EPFL Embedded Systems Laboratory)**
   - Référence : Orlandic et al., *The COUGHVID crowdsourcing dataset*, Scientific Data (2021).
   - Zenodo DOI : [10.5281/zenodo.7024894](https://zenodo.org/records/7024894)
   - Contient des enregistrements de toux annotés par des médecins pneumologues.

2. **Coswara Dataset (Indian Institute of Science - IISc Bangalore)**
   - Référence : Bhattacharya et al., *Coswara — A Database of In-the-wild Audio Samples for COVID-19 Detection*, Interspeech (2021).
   - Dépôt officiel : [https://github.com/iiscleap/Coswara-Data](https://github.com/iiscleap/Coswara-Data)
   - Contient des enregistrements de toux lourde (`cough-heavy.wav`) et légère (`cough-shallow.wav`) avec statut clinique vérifié et identifiants de participants (`id`).

## Prévention des Fuites de Données (Data Leakage)
Le découpage (Train / Val / Test) est rigoureusement effectué **par participant unique** (`participant_id`). Ainsi, aucune toux provenant d'un même individu ne peut se trouver simultanément dans les ensembles d'entraînement et de test.
