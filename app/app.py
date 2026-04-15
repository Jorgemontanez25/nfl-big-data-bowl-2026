import sys
from pathlib import Path
import json
import joblib
import gradio as gr

# --------------------------------------------------
# Make sure the project root is in the Python path
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.deploy_features import prepare_features_for_inference

# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------
MODEL_DIR = BASE_DIR / "models" / "final_deploy_xgboost"

MODEL_X_PATH = MODEL_DIR / "model_ball_land_x.pkl"
MODEL_Y_PATH = MODEL_DIR / "model_ball_land_y.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

model_x = joblib.load(MODEL_X_PATH)
model_y = joblib.load(MODEL_Y_PATH)

with open(FEATURES_PATH, "r") as f:
    trained_feature_columns = json.load(f)

# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_ball_landing(
    x,
    y,
    s,
    a,
    direction,
    orientation,
    absolute_yardline_number,
    player_weight,
    player_height_inches,
    player_age,
    play_direction,
    player_position,
    player_side,
    player_role,
    x_prev,
    y_prev,
    s_prev,
    a_prev,
):
    features = prepare_features_for_inference(
        trained_feature_columns=trained_feature_columns,
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

    pred_x = float(model_x.predict(features)[0])
    pred_y = float(model_y.predict(features)[0])

    return {
        "predicted_ball_land_x": round(pred_x, 2),
        "predicted_ball_land_y": round(pred_y, 2),
    }

# --------------------------------------------------
# Interface
# --------------------------------------------------
title = "🏈 NFL Ball Landing Predictor"
description = """
Predict where the ball will land using player tracking features and a deploy-safe XGBoost model.

This demo uses:
- player movement and direction
- contextual football features
- engineered spatial and temporal signals

Output:
- predicted ball landing x coordinate
- predicted ball landing y coordinate
"""

demo = gr.Interface(
    fn=predict_ball_landing,
    inputs=[
        gr.Number(label="x (current player x position)", value=45.0),
        gr.Number(label="y (current player y position)", value=20.0),
        gr.Number(label="speed s", value=5.0),
        gr.Number(label="acceleration a", value=1.0),
        gr.Number(label="direction dir (degrees)", value=90.0),
        gr.Number(label="orientation o (degrees)", value=90.0),
        gr.Number(label="absolute_yardline_number", value=50.0),
        gr.Number(label="player_weight", value=200.0),
        gr.Number(label="player_height_inches", value=72.0),
        gr.Number(label="player_age", value=25.0),
        gr.Radio(
            choices=["left", "right"],
            label="play_direction",
            value="right"
        ),
        gr.Dropdown(
            choices=["QB", "RB", "FB", "WR", "TE", "C", "G", "T", "CB", "S", "LB", "DL", "DB"],
            label="player_position",
            value="WR"
        ),
        gr.Dropdown(
            choices=["offense", "defense", "unknown"],
            label="player_side",
            value="offense"
        ),
        gr.Dropdown(
            choices=["offense", "defense", "coverage", "unknown"],
            label="player_role",
            value="offense"
        ),
        gr.Number(label="x_prev (previous x)", value=44.0),
        gr.Number(label="y_prev (previous y)", value=19.5),
        gr.Number(label="s_prev (previous speed)", value=4.8),
        gr.Number(label="a_prev (previous acceleration)", value=0.8),
    ],
    outputs=gr.JSON(label="Prediction"),
    title=title,
    description=description,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
