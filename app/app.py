with gr.Blocks(title="NFL Ball Landing Predictor") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            if LOGO_PATH.exists():
                gr.Image(
                    value=str(LOGO_PATH),
                    show_label=False,
                    container=False,
                    width=110,
                    height=110
                )

            gr.Markdown(
                """
                <div style="text-align: center;">
                    <h1 style="margin-bottom: 0.2rem;">NFL Ball Landing Predictor</h1>
                    <p style="margin-top: 0;">
                        Predict where the ball will land using deploy-safe engineered features and an XGBoost model.
                    </p>
                </div>
                """
            )
