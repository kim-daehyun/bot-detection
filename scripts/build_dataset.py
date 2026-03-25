from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================
# 1) 경로 설정
# =========================
FE_INPUT_PATH = Path("./data/FE/feature/[over_sampling]fe_preprocess.csv")
FE_TRAIN_PATH = Path("./data/FE/dataset/trainset/fe_trainset.csv")
FE_VALID_PATH = Path("./data/FE/dataset/validset/fe_validset.csv")
FE_TEST_PATH = Path("./data/FE/dataset/testset/fe_testset.csv")

BE_INPUT_PATH = Path("./data/BE/feature/[over_sampling]be_preprocess.csv")
BE_TRAIN_PATH = Path("./data/BE/dataset/trainset/be_trainset.csv")
BE_VALID_PATH = Path("./data/BE/dataset/validset/be_validset.csv")
BE_TEST_PATH = Path("./data/BE/dataset/testset/be_testset.csv")


# =========================
# 2) split 비율
# =========================
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_STATE = 42


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_input_df(df: pd.DataFrame, input_path: Path) -> None:
    if df.empty:
        raise ValueError(f"[ERROR] Input file is empty: {input_path}")

    if "label" not in df.columns:
        raise ValueError(f"[ERROR] 'label' column not found in: {input_path}")

    if df["label"].isna().any():
        raise ValueError(f"[ERROR] 'label' column contains NaN values in: {input_path}")

    unique_labels = df["label"].nunique()
    if unique_labels < 2:
        raise ValueError(
            f"[ERROR] Need at least 2 label classes for stratified split: {input_path}"
        )

    min_class_count = df["label"].value_counts().min()
    if min_class_count < 2:
        raise ValueError(
            f"[ERROR] Each label needs at least 2 samples for stratified split: {input_path}"
        )


def print_label_distribution(name: str, df: pd.DataFrame) -> None:
    total = len(df)
    dist = df["label"].value_counts(normalize=True).sort_index().to_dict()
    print(f"[INFO] {name}: rows={total}, label_ratio={dist}")


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    8:1:1 split
    - 먼저 train 80 / temp 20
    - temp 20을 valid 10 / test 10으로 반반 분할
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=RANDOM_STATE,
        stratify=df["label"],
        shuffle=True,
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
        shuffle=True,
    )

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False, encoding="utf-8")


def process_one_dataset(
    name: str,
    input_path: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"[ERROR] Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_input_df(df, input_path)

    print(f"\n=== Processing {name} dataset ===")
    print_label_distribution(f"{name} input", df)

    train_df, valid_df, test_df = split_dataset(df)

    print_label_distribution(f"{name} train", train_df)
    print_label_distribution(f"{name} valid", valid_df)
    print_label_distribution(f"{name} test", test_df)

    save_dataset(train_df, train_path)
    save_dataset(valid_df, valid_path)
    save_dataset(test_df, test_path)

    print(f"[DONE] {name} train saved -> {train_path}")
    print(f"[DONE] {name} valid saved -> {valid_path}")
    print(f"[DONE] {name} test saved  -> {test_path}")


def main() -> None:
    process_one_dataset(
        name="FE",
        input_path=FE_INPUT_PATH,
        train_path=FE_TRAIN_PATH,
        valid_path=FE_VALID_PATH,
        test_path=FE_TEST_PATH,
    )

    process_one_dataset(
        name="BE",
        input_path=BE_INPUT_PATH,
        train_path=BE_TRAIN_PATH,
        valid_path=BE_VALID_PATH,
        test_path=BE_TEST_PATH,
    )

    print("\n[DONE] All dataset splits completed.")


if __name__ == "__main__":
    main()