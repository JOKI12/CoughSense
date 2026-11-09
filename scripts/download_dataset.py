"""
Script d'acquisition et de téléchargement des jeux de données officiels pour CoughSense.

Sources officielles prises en charge :
1. Coswara Dataset (IISc Bangalore) : https://github.com/iiscleap/Coswara-Data
2. COUGHVID Dataset (EPFL ESL) : https://doi.org/10.5281/zenodo.7024894

Ce script télécharge les archives directement depuis les dépôts publics officiels,
extrait les fichiers audio de toux réels ('cough-heavy.wav', 'cough-shallow.wav')
ainsi que leurs métadonnées associées dans 'data/raw/'.
"""

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request
import zipfile

import pandas as pd
from tqdm import tqdm


RAW_DIR = Path("data/raw")
COSWARA_DIR = RAW_DIR / "coswara"
COUGHVID_DIR = RAW_DIR / "coughvid"

COSWARA_BASE_URL = "https://raw.githubusercontent.com/iiscleap/Coswara-Data/master"
EPFL_COUGHVID_RAW_BASE = "https://raw.githubusercontent.com/esl-epfl/COUGHVID/main"


class DownloadProgressBar(tqdm):
    """Barre de progression pour urllib.request.urlretrieve."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: Path) -> Path:
    """Télécharge un fichier distant vers le disque avec suivi de progression."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> Téléchargement : {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "CoughSense-SignalProcessing/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))
    return dest_path


def download_coswara_metadata() -> Path:
    """Télécharge le fichier global de métadonnées de Coswara."""
    COSWARA_DIR.mkdir(parents=True, exist_ok=True)
    meta_url = f"{COSWARA_BASE_URL}/combined_data.csv"
    dest_path = COSWARA_DIR / "combined_data.csv"
    if not dest_path.exists():
        download_file(meta_url, dest_path)
    else:
        print(f"Métadonnées Coswara déjà présentes : {dest_path}")
    return dest_path


def download_and_extract_coswara_batch(batch_date: str) -> None:
    """
    Télécharge et décompresse un lot spécifique du jeu de données Coswara.
    Extrait les fichiers 'cough-heavy.wav' et 'cough-shallow.wav' par participant.
    """
    batch_dir = COSWARA_DIR / batch_date
    batch_dir.mkdir(parents=True, exist_ok=True)

    # 1. Télécharger le fichier CSV du lot
    csv_url = f"{COSWARA_BASE_URL}/{batch_date}/{batch_date}.csv"
    csv_dest = batch_dir / f"{batch_date}.csv"
    try:
        if not csv_dest.exists():
            download_file(csv_url, csv_dest)
    except Exception as e:
        print(f"Avertissement : impossible de télécharger {csv_url} ({e})")

    # 2. Télécharger l'archive audio tar.gz (.aa)
    tar_filename = f"{batch_date}.tar.gz.aa"
    tar_url = f"{COSWARA_BASE_URL}/{batch_date}/{tar_filename}"
    tar_dest = batch_dir / tar_filename

    if not tar_dest.exists():
        download_file(tar_url, tar_dest)

    # 3. Extraction ciblée des fichiers audio de toux et métadonnées
    print(f"-> Extraction des enregistrements de toux pour le lot {batch_date}...")
    extracted_count = 0
    with tarfile.open(tar_dest, mode="r:gz") as tar:
        for member in tar.getmembers():
            # Ne décompresser que les toux et métadonnées pour économiser l'espace disque
            if member.name.endswith("cough-heavy.wav") or member.name.endswith("cough-shallow.wav") or member.name.endswith("metadata.json"):
                tar.extract(member, path=COSWARA_DIR)
                extracted_count += 1

    print(f"-> Extraction terminée : {extracted_count} fichiers extraits dans {COSWARA_DIR / batch_date}")


def download_coughvid_samples() -> None:
    """Télécharge les enregistrements officiels de référence du projet EPFL COUGHVID."""
    COUGHVID_DIR.mkdir(parents=True, exist_ok=True)
    sample_files = ["cough.wav", "not_cough.wav"]

    for fname in sample_files:
        url = f"{EPFL_COUGHVID_RAW_BASE}/sample_recordings/{fname}"
        dest = COUGHVID_DIR / fname
        if not dest.exists():
            download_file(url, dest)
        else:
            print(f"Échantillon COUGHVID déjà présent : {dest}")


def main():
    parser = argparse.ArgumentParser(description="Téléchargement des jeux de données réels pour CoughSense.")
    parser.add_argument(
        "--source",
        choices=["coswara", "coughvid_samples", "all"],
        default="all",
        help="Source de données à acquérir (coswara, coughvid_samples ou all)"
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        default=["20200911", "20200505"],
        help="Identifiants des lots Coswara à télécharger (ex: 20200911 20200505 20200824)"
    )
    args = parser.parse_args()

    print("==================================================================")
    print(" CoughSense - Acquisition des Enregistrements Audio Réels")
    print("==================================================================")

    if args.source in ["coughvid_samples", "all"]:
        print("\n--- 1. Échantillons officiels EPFL COUGHVID ---")
        download_coughvid_samples()

    if args.source in ["coswara", "all"]:
        print("\n--- 2. Jeu de données officiel IISc Coswara ---")
        download_coswara_metadata()
        for batch in args.batches:
            print(f"\nTraitement du lot Coswara : {batch}")
            download_and_extract_coswara_batch(batch)

    print("\n==================================================================")
    print(" Téléchargement et extraction terminés avec succès !")
    print(f" Répertoire des données brutes : {RAW_DIR.resolve()}")
    print("==================================================================")


if __name__ == "__main__":
    main()
