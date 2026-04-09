from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/lstm/lstm_panel.csv")
OUTPUT_DIR = Path("data/processed/lstm")

SEQUENCE_LENGTH = 30
TRAIN_END_DATE = "2022-12-31"
VALID_END_DATE = "2023-12-31"

# Keep all positive sequences, but downsample negative-only sequences in train/valid.
TRAIN_NEGATIVE_RATIO = 3.0
VALID_NEGATIVE_RATIO = 3.0
TEST_NEGATIVE_RATIO = None  # keep full test set

RANDOM_SEED = 42


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"segment_id", "date", "crash_occurred"}
    return [col for col in df.columns if col not in excluded]


def split_segment_dataframe(group: pd.DataFrame):
    train_df = group[group["date"] <= pd.Timestamp(TRAIN_END_DATE)].copy()
    valid_df = group[
        (group["date"] > pd.Timestamp(TRAIN_END_DATE))
        & (group["date"] <= pd.Timestamp(VALID_END_DATE))
    ].copy()
    test_df = group[group["date"] > pd.Timestamp(VALID_END_DATE)].copy()

    return train_df, valid_df, test_df


def build_split_sequences(
    split_df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
):
    """
    Build sequences within a single split only.
    Each sequence uses previous `sequence_length` rows to predict current row.
    """
    if len(split_df) <= sequence_length:
        return [], [], []

    features = split_df[feature_cols].to_numpy(dtype=np.float32)
    targets = split_df["crash_occurred"].to_numpy(dtype=np.int64)
    dates = split_df["date"].to_numpy()

    X_list = []
    y_list = []
    meta_list = []

    for end_idx in range(sequence_length, len(split_df)):
        start_idx = end_idx - sequence_length

        X_seq = features[start_idx:end_idx]
        y_target = targets[end_idx]
        target_date = pd.Timestamp(dates[end_idx])

        X_list.append(X_seq)
        y_list.append(y_target)
        meta_list.append(target_date)

    return X_list, y_list, meta_list


def downsample_negatives(X, y, meta_dates, negative_ratio, rng):
    """
    Keep all positives. Keep at most `negative_ratio * positives` negatives.
    """
    if negative_ratio is None:
        return X, y, meta_dates

    y = np.asarray(y)
    positive_idx = np.where(y == 1)[0]
    negative_idx = np.where(y == 0)[0]

    n_pos = len(positive_idx)
    n_neg_keep = int(n_pos * negative_ratio)

    if n_pos == 0:
        # No positives in this chunk; keep a small random negative sample to avoid emptiness
        n_neg_keep = min(len(negative_idx), 5000)

    if len(negative_idx) > n_neg_keep:
        keep_neg = rng.choice(negative_idx, size=n_neg_keep, replace=False)
    else:
        keep_neg = negative_idx

    keep_idx = np.concatenate([positive_idx, keep_neg])
    keep_idx.sort()

    X_kept = [X[i] for i in keep_idx]
    y_kept = [y[i] for i in keep_idx]
    meta_kept = [meta_dates[i] for i in keep_idx]

    return X_kept, y_kept, meta_kept


def save_split(X_list, y_list, meta_df, split_name: str, feature_cols: list[str]):
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.save(OUTPUT_DIR / f"X_{split_name}.npy", X)
    np.save(OUTPUT_DIR / f"y_{split_name}.npy", y)
    meta_df.to_csv(OUTPUT_DIR / f"meta_{split_name}.csv", index=False)

    print(f"Saved {split_name}: X shape={X.shape}, y shape={y.shape}")

    return {
        f"{split_name}_shape": list(X.shape),
        f"{split_name}_positive_rate": float(y.mean()) if len(y) else None,
    }


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading LSTM panel...")
    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["segment_id"] = df["segment_id"].astype(str)
    df = df[df["date"].notna()].copy()
    df = df.sort_values(["segment_id", "date"]).reset_index(drop=True)

    feature_cols = get_feature_columns(df)

    print(f"Panel rows: {len(df):,}")
    print(f"Segments: {df['segment_id'].nunique():,}")
    print(f"Feature count: {len(feature_cols)}")
    print("Feature columns:")
    print(feature_cols)

    train_X, train_y, train_meta = [], [], []
    valid_X, valid_y, valid_meta = [], [], []
    test_X, test_y, test_meta = [], [], []

    grouped = df.groupby("segment_id", sort=False)

    print(f"\nBuilding {SEQUENCE_LENGTH}-day sequences...")

    for i, (segment_id, group) in enumerate(grouped, start=1):
        group = group.sort_values("date").reset_index(drop=True)

        train_df, valid_df, test_df = split_segment_dataframe(group)

        X_tmp, y_tmp, d_tmp = build_split_sequences(train_df, feature_cols, SEQUENCE_LENGTH)
        if X_tmp:
            X_tmp, y_tmp, d_tmp = downsample_negatives(
                X_tmp, y_tmp, d_tmp, TRAIN_NEGATIVE_RATIO, rng
            )
            train_X.extend(X_tmp)
            train_y.extend(y_tmp)
            train_meta.extend([(segment_id, d) for d in d_tmp])

        X_tmp, y_tmp, d_tmp = build_split_sequences(valid_df, feature_cols, SEQUENCE_LENGTH)
        if X_tmp:
            X_tmp, y_tmp, d_tmp = downsample_negatives(
                X_tmp, y_tmp, d_tmp, VALID_NEGATIVE_RATIO, rng
            )
            valid_X.extend(X_tmp)
            valid_y.extend(y_tmp)
            valid_meta.extend([(segment_id, d) for d in d_tmp])

        X_tmp, y_tmp, d_tmp = build_split_sequences(test_df, feature_cols, SEQUENCE_LENGTH)
        if X_tmp:
            # Keep full test set by default
            if TEST_NEGATIVE_RATIO is not None:
                X_tmp, y_tmp, d_tmp = downsample_negatives(
                    X_tmp, y_tmp, d_tmp, TEST_NEGATIVE_RATIO, rng
                )
            test_X.extend(X_tmp)
            test_y.extend(y_tmp)
            test_meta.extend([(segment_id, d) for d in d_tmp])

        if i % 1000 == 0:
            print(
                f"Processed {i:,} segments | "
                f"train seq: {len(train_X):,} | "
                f"valid seq: {len(valid_X):,} | "
                f"test seq: {len(test_X):,}"
            )

    print("\nBuilding metadata tables...")
    meta_train = pd.DataFrame(train_meta, columns=["segment_id", "target_date"])
    meta_valid = pd.DataFrame(valid_meta, columns=["segment_id", "target_date"])
    meta_test = pd.DataFrame(test_meta, columns=["segment_id", "target_date"])

    print("\nSaving arrays...")
    summary = {
        "sequence_length": SEQUENCE_LENGTH,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "train_negative_ratio": TRAIN_NEGATIVE_RATIO,
        "valid_negative_ratio": VALID_NEGATIVE_RATIO,
        "test_negative_ratio": TEST_NEGATIVE_RATIO,
    }

    summary.update(save_split(train_X, train_y, meta_train, "train", feature_cols))
    summary.update(save_split(valid_X, valid_y, meta_valid, "valid", feature_cols))
    summary.update(save_split(test_X, test_y, meta_test, "test", feature_cols))

    joblib.dump(feature_cols, OUTPUT_DIR / "feature_columns.joblib")

    with open(OUTPUT_DIR / "sequence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved feature columns to {OUTPUT_DIR / 'feature_columns.joblib'}")
    print(f"Saved summary to {OUTPUT_DIR / 'sequence_summary.json'}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()