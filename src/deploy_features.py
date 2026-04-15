from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEPLOY_CATEGORICAL_COLS = [
    "player_position",
    "player_side",
    "player_role",
]


DEPLOY_RAW_FEATURE_COLS = [
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
    "speed_x",
    "speed_y",
    "acc_x",
    "acc_y",
    "dir_minus_o",
    "momentum",
    "momentum_x",
    "momentum_y",
    "x_centered",
    "y_centered",
    "dist_from_field_center",
    "x_prev",
    "y_prev",
    "s_prev",
    "a_prev",
    "delta_x",
    "delta_y",
    "delta_s",
    "delta_a",
    "speed_bin_code",
]


def compute_is_moving_right(play_direction: str) -> int:
    return int(str(play_direction).strip().lower() == "right")


def normalize_angle_diff(angle_1: float, angle_2: float) -> float:
    diff = angle_1 - angle_2
    return ((diff + 180) % 360) - 180


def compute_speed_components(speed: float, direction_deg: float) -> tuple[float, float]:
    direction_rad = np.deg2rad(direction_deg)
    speed_x = speed * np.cos(direction_rad)
    speed_y = speed * np.sin(direction_rad)
    return float(speed_x), float(speed_y)


def compute_acc_components(acceleration: float, direction_deg: float) -> tuple[float, float]:
    direction_rad = np.deg2rad(direction_deg)
    acc_x = acceleration * np.cos(direction_rad)
    acc_y = acceleration * np.sin(direction_rad)
    return float(acc_x), float(acc_y)


def compute_speed_bin_code(speed: float) -> int:
    """
    Simple deploy-time binning for speed.
    These bins are hand-defined so the app does not depend on qcut at inference time.
    """
    if speed < 1.0:
        return 0
    if speed < 3.0:
        return 1
    if speed < 5.0:
        return 2
    if speed < 7.0:
        return 3
    return 4


def build_deploy_feature_dict(
    x: float,
    y: float,
    s: float,
    a: float,
    direction: float,
    orientation: float,
    absolute_yardline_number: float,
    player_weight: float,
    player_height_inches: float,
    player_age: float,
    play_direction: str,
    player_position: str,
    player_side: str,
    player_role: str,
    x_prev: float = 0.0,
    y_prev: float = 0.0,
    s_prev: float = 0.0,
    a_prev: float = 0.0,
) -> dict[str, Any]:
    speed_x, speed_y = compute_speed_components(s, direction)
    acc_x, acc_y = compute_acc_components(a, direction)

    is_moving_right = compute_is_moving_right(play_direction)
    dir_minus_o = normalize_angle_diff(direction, orientation)

    momentum = player_weight * s
    momentum_x = player_weight * speed_x
    momentum_y = player_weight * speed_y

    x_centered = x - 60.0
    y_centered = y - 26.65
    dist_from_field_center = float(np.sqrt(x_centered**2 + y_centered**2))

    delta_x = x - x_prev
    delta_y = y - y_prev
    delta_s = s - s_prev
    delta_a = a - a_prev

    speed_bin_code = compute_speed_bin_code(s)

    return {
        "x": x,
        "y": y,
        "s": s,
        "a": a,
        "dir": direction,
        "o": orientation,
        "absolute_yardline_number": absolute_yardline_number,
        "player_weight": player_weight,
        "player_height_inches": player_height_inches,
        "player_age": player_age,
        "is_moving_right": is_moving_right,
        "player_position": player_position,
        "player_side": player_side,
        "player_role": player_role,
        "speed_x": speed_x,
        "speed_y": speed_y,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "dir_minus_o": dir_minus_o,
        "momentum": momentum,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
        "x_centered": x_centered,
        "y_centered": y_centered,
        "dist_from_field_center": dist_from_field_center,
        "x_prev": x_prev,
        "y_prev": y_prev,
        "s_prev": s_prev,
        "a_prev": a_prev,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "delta_s": delta_s,
        "delta_a": delta_a,
        "speed_bin_code": speed_bin_code,
    }


def build_deploy_feature_frame(
    x: float,
    y: float,
    s: float,
    a: float,
    direction: float,
    orientation: float,
    absolute_yardline_number: float,
    player_weight: float,
    player_height_inches: float,
    player_age: float,
    play_direction: str,
    player_position: str,
    player_side: str,
    player_role: str,
    x_prev: float = 0.0,
    y_prev: float = 0.0,
    s_prev: float = 0.0,
    a_prev: float = 0.0,
) -> pd.DataFrame:
    row = build_deploy_feature_dict(
        x=x,
        y=y,
        s=s,
        a=a,
        direction=direction,
        orientation=orientation,
        absolute_yardline_number=absolute_yardline_number,
        player_weight=player_weight,
        player_height_inches=player_height_inches,
        player_age=player_age,
        play_direction=play_direction,
        player_position=player_position,
        player_side=player_side,
        player_role=player_role,
        x_prev=x_prev,
        y_prev=y_prev,
        s_prev=s_prev,
        a_prev=a_prev,
    )

    df = pd.DataFrame([row])
    return df


def encode_deploy_features(
    df: pd.DataFrame,
    trained_feature_columns: list[str],
) -> pd.DataFrame:
    df_encoded = pd.get_dummies(
        df,
        columns=DEPLOY_CATEGORICAL_COLS,
        dummy_na=True,
    )

    df_encoded = df_encoded.reindex(columns=trained_feature_columns, fill_value=0)
    return df_encoded


def load_trained_feature_columns(features_path: str | Path) -> list[str]:
    features_path = Path(features_path)
    with open(features_path, "r") as f:
        return json.load(f)


def prepare_features_for_inference(
    trained_feature_columns: list[str],
    x: float,
    y: float,
    s: float,
    a: float,
    direction: float,
    orientation: float,
    absolute_yardline_number: float,
    player_weight: float,
    player_height_inches: float,
    player_age: float,
    play_direction: str,
    player_position: str,
    player_side: str,
    player_role: str,
    x_prev: float = 0.0,
    y_prev: float = 0.0,
    s_prev: float = 0.0,
    a_prev: float = 0.0,
) -> pd.DataFrame:
    raw_df = build_deploy_feature_frame(
        x=x,
        y=y,
        s=s,
        a=a,
        direction=direction,
        orientation=orientation,
        absolute_yardline_number=absolute_yardline_number,
        player_weight=player_weight,
        player_height_inches=player_height_inches,
        player_age=player_age,
        play_direction=play_direction,
        player_position=player_position,
        player_side=player_side,
        player_role=player_role,
        x_prev=x_prev,
        y_prev=y_prev,
        s_prev=s_prev,
        a_prev=a_prev,
    )

    encoded_df = encode_deploy_features(raw_df, trained_feature_columns)
    return encoded_df