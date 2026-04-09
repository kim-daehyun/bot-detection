from __future__ import annotations

from pathlib import Path
import warnings
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


TRAIN_PATH = Path("./data/FE/dataset/trainset/fe_trainset.csv")
VALID_PATH = Path("./data/FE/dataset/validset/fe_validset.csv")
TEST_PATH = Path("./data/FE/dataset/testset/fe_testset.csv")
MODEL_DIR = Path("./model/FE")

TARGET_COL = "label"

# 학습에 쓰지 않는 컬럼
DROP_COLS = [
    "source_file",
]

# 실제 학습에 사용할 feature 3개
FEATURE_COLS = [
    "duration_ms",
    "mousemove_teleport_count",
    "mousemove_count",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Dataset not found: {path}")
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"[ERROR] '{TARGET_COL}' column not found.")

    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] Missing feature columns: {missing_cols}")

    y = df[TARGET_COL].astype(int)
    x = df[FEATURE_COLS].copy()

    return x, y


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = FEATURE_COLS

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
        ]
    )


def build_model_dict() -> dict[str, object]:
    """
    가능한 모델만 등록.
    xgboost / catboost가 환경 문제로 로드 실패하면 경고만 출력하고 계속 진행.
    """
    model_dict: dict[str, object] = {}

    model_dict["random_forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model_dict["decision_tree"] = DecisionTreeClassifier(
        random_state=42,
        max_depth=None,
    )

    try:
        from xgboost import XGBClassifier

        model_dict["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        print("[OK] xgboost loaded successfully.")
    except Exception as e:
        print(f"[WARN] xgboost unavailable. skipped. reason={e}")

    try:
        from catboost import CatBoostClassifier

        model_dict["catboost"] = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=0,
            random_state=42,
        )
        print("[OK] catboost loaded successfully.")
    except Exception as e:
        print(f"[WARN] catboost unavailable. skipped. reason={e}")

    return model_dict


def train_and_evaluate(
    model_name: str,
    estimator,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> dict:
    print(f"\n[START] Training FE model: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(x_train, y_train)
    print(f"[DONE] Training finished: {model_name}")

    valid_pred = pipeline.predict(x_valid)

    precision = precision_score(y_valid, valid_pred, zero_division=0)
    recall = recall_score(y_valid, valid_pred, zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump(pipeline, model_path)
    print(f"[SAVE] Model saved -> {model_path}")

    return {
        "model_name": model_name,
        "precision": precision,
        "recall": recall,
        "score_avg": (precision + recall) / 2.0,
        "model_path": str(model_path),
    }


def main() -> None:
    print("=== FE model training started ===")

    train_df = load_dataset(TRAIN_PATH)
    valid_df = load_dataset(VALID_PATH)
    _ = load_dataset(TEST_PATH)

    x_train, y_train = split_xy(train_df)
    x_valid, y_valid = split_xy(valid_df)

    print(f"[INFO] FE train shape: {x_train.shape}, valid shape: {x_valid.shape}")
    print(f"[INFO] FE feature columns: {FEATURE_COLS}")
    print(f"[INFO] Label meaning: 0=human, 1=bot")

    preprocessor = build_preprocessor(x_train)
    model_dict = build_model_dict()

    if not model_dict:
        raise RuntimeError("[ERROR] No available models to train.")

    results = []
    total = len(model_dict)

    for idx, (model_name, estimator) in enumerate(model_dict.items(), start=1):
        print(f"\n[PROGRESS] FE {idx}/{total}: {model_name}")
        result = train_and_evaluate(
            model_name=model_name,
            estimator=estimator,
            preprocessor=preprocessor,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
        )
        results.append(result)

    result_df = pd.DataFrame(results).sort_values(
        by=["score_avg", "precision", "recall"],
        ascending=False,
    )

    print("\n=== FE validation performance ===")
    print(
        result_df[
            ["model_name", "precision", "recall", "score_avg", "model_path"]
        ].to_string(index=False)
    )

    best_row = result_df.iloc[0]
    print(
        f"\n[BEST FE MODEL] {best_row['model_name']} "
        f"(precision={best_row['precision']:.4f}, recall={best_row['recall']:.4f})"
    )

    print("\n=== FE model training completed ===")


if __name__ == "__main__":
    main()