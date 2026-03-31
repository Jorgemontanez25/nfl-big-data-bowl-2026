import pandas as pd
import numpy as np


def height_to_inches(height_str: str) -> float:
    if pd.isna(height_str):
        return np.nan
    try:
        feet, inches = str(height_str).split("-")
        return int(feet) * 12 + int(inches)
    except Exception:
        return np.nan


def compute_player_age(df: pd.DataFrame, reference_year: int = 2023) -> pd.Series:
    birth_dates = pd.to_datetime(df["player_birth_date"], errors="coerce")
    return reference_year - birth_dates.dt.year


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["player_height_inches"] = df["player_height"].apply(height_to_inches)
    df["player_age"] = compute_player_age(df)
    df["is_moving_right"] = (df["play_direction"] == "right").astype(int)

    return df


def select_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["player_to_predict"] == True].copy()


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "game_id",
        "play_id",
        "nfl_id",
        "frame_id",
        "x",
        "y",
        "s",
        "a",
        "dir",
        "o",
        "absolute_yardline_number",
        "player_weight",
        "player_height_inches",
        "player_age",
        "is_moving_right",
        "player_position",
        "player_side",
        "player_role",
        "ball_land_x",
        "ball_land_y",
        "num_frames_output",
    ]
    return df[columns].copy()