
from pathlib import Path
import pandas as pd

from src.config import TRAIN_DIR, TEST_PATH, TEST_INPUT_PATH


def list_train_files(train_dir: Path = TRAIN_DIR):
    files = sorted(train_dir.glob("input_2023_w*.csv"))
    if not files:
        raise FileNotFoundError(f"No training files found in {train_dir}")
    return files


def load_train_week(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


def load_test_input() -> pd.DataFrame:
    return pd.read_csv(TEST_INPUT_PATH)


def load_train_sample(n_files: int = 1) -> pd.DataFrame:
    files = list_train_files()[:n_files]
    dfs = [load_train_week(f) for f in files]
    return pd.concat(dfs, ignore_index=True)