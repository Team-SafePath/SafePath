from pathlib import Path
import json
import random

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("data/processed/lstm")
MODEL_DIR = Path("models")

X_TRAIN_PATH = DATA_DIR / "X_train.npy"
Y_TRAIN_PATH = DATA_DIR / "y_train.npy"
X_VALID_PATH = DATA_DIR / "X_valid.npy"
Y_VALID_PATH = DATA_DIR / "y_valid.npy"
X_TEST_PATH = DATA_DIR / "X_test.npy"
Y_TEST_PATH = DATA_DIR / "y_test.npy"

FEATURE_COLUMNS_PATH = DATA_DIR / "feature_columns.joblib"
SUMMARY_PATH = DATA_DIR / "sequence_summary.json"

MODEL_PATH = MODEL_DIR / "lstm_model.pt"
METRICS_PATH = MODEL_DIR / "lstm_metrics.json"

RANDOM_SEED = 42
BATCH_SIZE = 512
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 15
PATIENCE = 4


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CrashLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        logits = self.head(last_hidden).squeeze(-1)
        return logits


def load_data():
    print("Loading LSTM sequence data...")

    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)

    X_valid = np.load(X_VALID_PATH)
    y_valid = np.load(Y_VALID_PATH)

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print(f"Train shape: {X_train.shape}")
    print(f"Valid shape: {X_valid.shape}")
    print(f"Test shape:  {X_test.shape}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_columns, summary


def create_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_valid_t = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_t = torch.tensor(y_valid, dtype=torch.float32)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        TensorDataset(X_valid_t, y_valid_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, valid_loader, test_loader


def compute_pos_weight(y_train: np.ndarray) -> float:
    positives = float((y_train == 1).sum())
    negatives = float((y_train == 0).sum())

    if positives == 0:
        return 1.0

    return negatives / positives


def predict_probabilities(model, data_loader, device):
    model.eval()

    all_probs = []
    all_targets = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            batch_size = y_batch.size(0)
            total_count += batch_size

            all_probs.append(probs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets).astype(int)

    return y_true, y_prob


def evaluate_with_threshold(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def tune_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.951, 0.05)

    best_threshold = 0.5
    best_score = -np.inf
    best_metrics = None

    for threshold in thresholds:
        metrics = evaluate_with_threshold(y_true, y_prob, threshold)
        score = metrics["f1"]

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def train_one_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()

    running_loss = 0.0
    total_count = 0

    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = y_batch.size(0)
        running_loss += loss.item() * batch_size
        total_count += batch_size

    return running_loss / max(total_count, 1)


def compute_loss(model, data_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def main():
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        feature_columns,
        summary,
    ) = load_data()

    train_loader, valid_loader, test_loader = create_dataloaders(
        X_train, y_train, X_valid, y_valid, X_test, y_test
    )

    input_size = X_train.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_weight_value = compute_pos_weight(y_train)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    print(f"Using device: {device}")
    print(f"Input size: {input_size}")
    print(f"Sequence length: {X_train.shape[1]}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Training pos_weight: {pos_weight_value:.4f}")

    model = CrashLSTM(input_size=input_size).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_valid_auc = -np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    print("\nStarting training...")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss = compute_loss(model, valid_loader, loss_fn, device)
        y_valid_true, y_valid_prob = predict_probabilities(model, valid_loader, device)

        valid_auc = roc_auc_score(y_valid_true, y_valid_prob)
        valid_ap = average_precision_score(y_valid_true, y_valid_prob)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_roc_auc": valid_auc,
            "valid_average_precision": valid_ap,
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_loss:.4f} | "
            f"valid_roc_auc={valid_auc:.4f} | "
            f"valid_ap={valid_ap:.4f}"
        )

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "feature_columns": feature_columns,
                    "config": {
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "sequence_length": summary.get("sequence_length"),
                    },
                },
                MODEL_PATH,
            )
            print(f"Saved new best model to {MODEL_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"\nBest validation ROC-AUC: {best_valid_auc:.4f} at epoch {best_epoch}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_valid_true, y_valid_prob = predict_probabilities(model, valid_loader, device)
    best_threshold, valid_metrics = tune_threshold(y_valid_true, y_valid_prob)

    print("\nValidation Results")
    print(f"ROC-AUC: {valid_metrics['roc_auc']:.4f}")
    print(f"Average Precision: {valid_metrics['average_precision']:.4f}")
    print(f"Balanced Accuracy: {valid_metrics['balanced_accuracy']:.4f}")
    print(f"F1: {valid_metrics['f1']:.4f}")
    print(f"Best Threshold: {best_threshold:.2f}")
    print("Confusion Matrix:")
    print(np.array(valid_metrics["confusion_matrix"]))

    y_test_true, y_test_prob = predict_probabilities(model, test_loader, device)
    test_metrics = evaluate_with_threshold(y_test_true, y_test_prob, best_threshold)

    print("\nTest Results")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Average Precision: {test_metrics['average_precision']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"Threshold: {best_threshold:.2f}")
    print("Confusion Matrix:")
    print(np.array(test_metrics["confusion_matrix"]))

    output = {
        "model": "LSTM",
        "best_epoch": best_epoch,
        "best_valid_roc_auc": best_valid_auc,
        "selected_threshold": best_threshold,
        "training_pos_weight": pos_weight_value,
        "config": {
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        },
        "data_summary": summary,
        "history": history,
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()