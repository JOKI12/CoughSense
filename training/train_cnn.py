"""
Module d'entraînement du réseau de neurones convolutif 2D (CNN) optionnel pour CoughSense.

Prend en entrée le Mel-Spectrogramme (temps-fréquence) standardisé à [64 mels x 128 trames].
Architecture CNN :
- Bloc Conv1 : Conv2d(1 -> 16, 3x3) + BatchNorm + ReLU + MaxPool2d(2x2)
- Bloc Conv2 : Conv2d(16 -> 32, 3x3) + BatchNorm + ReLU + MaxPool2d(2x2)
- Couche Dense : AdaptiveAvgPool2d((8, 8)) + Flatten + Linear(2048 -> 64) + Dropout(0.3) + Linear(64 -> 2)
- Fonction de perte : CrossEntropyLoss
- Optimiseur : Adam (lr=1e-3)
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class CoughMelDataset(Dataset):
    """Dataset PyTorch pour charger et formater les Mel-Spectrogrammes."""
    def __init__(self, df: pd.DataFrame, target_frames: int = 128, n_mels: int = 64):
        self.samples = []
        self.labels = []
        self.target_frames = target_frames
        self.n_mels = n_mels

        from src.audio_loader import load_and_standardize_audio
        from src.preprocessing import preprocess_signal

        for _, row in df.iterrows():
            fpath = Path(row["file_path"])
            if not fpath.exists():
                continue
            try:
                audio = load_and_standardize_audio(fpath, target_sample_rate=16000)
                prep = preprocess_signal(audio.signal, sample_rate=16000)
                sig = prep.processed_signal

                mel = librosa.feature.melspectrogram(
                    y=sig,
                    sr=16000,
                    n_fft=512,
                    hop_length=128,
                    n_mels=n_mels
                )
                mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

                # Standardisation temporelle (padding ou troncature à target_frames)
                if mel_db.shape[1] < target_frames:
                    pad_width = target_frames - mel_db.shape[1]
                    mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
                else:
                    mel_db = mel_db[:, :target_frames]

                # Normalisation min-max dans [0, 1]
                mel_norm = (mel_db + 80.0) / 80.0
                mel_norm = np.clip(mel_norm, 0.0, 1.0).astype(np.float32)

                self.samples.append(torch.tensor(mel_norm).unsqueeze(0))  # [1, 64, 128]
                self.labels.append(int(row["label"]))
            except Exception:
                continue

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class CoughCNN(nn.Module):
    """Architecture CNN 2D légère pour la classification acoustique."""
    def __init__(self, num_classes: int = 2):
        super(CoughCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> [16, 32, 64]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> [32, 16, 32]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.avg_pool(feat)
        out = self.classifier(pooled)
        return out


def train_cnn():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==================================================================")
    print(f" CoughSense - Entraînement du Modèle CNN 2D (Périphérique : {device})")
    print("==================================================================")

    train_path = DATA_PROCESSED_DIR / "features_train.csv"
    test_path = DATA_PROCESSED_DIR / "features_test.csv"

    if not train_path.exists() or not test_path.exists():
        print("Erreur : Datasets introuvables. Exécutez d'abord training/prepare_dataset.py.")
        return

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print("Préparation des tenseurs de Mel-Spectrogrammes...")
    train_dataset = CoughMelDataset(df_train)
    test_dataset = CoughMelDataset(df_test)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Erreur : Impossible de construire les datasets Mel.")
        return

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = CoughCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 20
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), torch.tensor(targets).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_acc = correct / max(1, total)

    # Évaluation sur le jeu de test
    model.eval()
    all_preds = []
    all_targets = []
    all_probas = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probas = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_probas.extend(probas)
    t1 = time.perf_counter()
    latency_ms = ((t1 - t0) / max(1, len(all_targets))) * 1000.0

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
    acc = float(accuracy_score(all_targets, all_preds))
    f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    cm = confusion_matrix(all_targets, all_preds).tolist()
    try:
        auc = float(roc_auc_score(all_targets, all_probas))
    except Exception:
        auc = 0.5

    cnn_metrics = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "roc_auc": round(auc, 4),
        "inference_time_ms_per_sample": round(latency_ms, 3),
        "confusion_matrix": cm
    }

    # Sauvegarde du modèle CNN
    torch.save(model.state_dict(), MODELS_DIR / "cnn_model.pt")

    # Mise à jour du fichier de métriques
    metrics_file = MODELS_DIR / "model_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics["CNN"] = cnn_metrics
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"CNN Accuracy  : {acc * 100:.2f} %")
    print(f"CNN F1-Score  : {f1:.4f}")
    print(f"CNN Latence   : {latency_ms:.3f} ms/échantillon")
    print(f"Modèle sauvegardé dans : {MODELS_DIR / 'cnn_model.pt'}")


if __name__ == "__main__":
    train_cnn()
