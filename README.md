# 🏈 NFL Ball Landing Prediction  
### Big Data Bowl 2026 | End-to-End ML System + Deployment

<p align="center">
  <img src="assets/nfl_logo.png" width="120"/>
</p>

<p align="center">
  <b>Predicting ball landing position from player tracking data using Machine Learning</b><br>
  Built with XGBoost · Deployed with Gradio · Hosted on Hugging Face Spaces
</p>

---

## 🚀 Live Demo

👉 **Try the App**  
https://huggingface.co/spaces/Montanez25/NFL_Player_Tracking_ML_Ball_Landing_Prediction

---

## 🧠 Project Overview

This project answers a key question in football analytics:

> **Given player tracking data at a specific moment, where will the ball land?**

Using NFL Big Data Bowl tracking data, this project builds a complete machine learning pipeline:

- Feature engineering from raw tracking data  
- Model development (Random Forest → XGBoost)  
- Evaluation and error analysis  
- Deployment as an interactive ML application  

---

## 🧩 End-to-End Pipeline

```
Raw Tracking Data
        ↓
Feature Engineering (movement + temporal + context)
        ↓
Model Training (Random Forest → XGBoost)
        ↓
Evaluation (MAE, RMSE, R² + error analysis)
        ↓
Deployment (Gradio + Hugging Face Spaces)
```

---

## ⚙️ Key Features

### 📊 Feature Engineering
- Player motion:
  - Speed (`s`)
  - Acceleration (`a`)
  - Direction & orientation
- Temporal dynamics:
  - Previous frame features (`x_prev`, `y_prev`, etc.)
- Context:
  - Player position, role, side
  - Play direction

---

### 🤖 Modeling

| Model            | Purpose        |
|------------------|--------------|
| Random Forest    | Baseline      |
| **XGBoost**      | Final Model   |

- Dual regression targets:
  - `ball_land_x`
  - `ball_land_y`

---

### 📈 Evaluation

- Metrics:
  - MAE
  - RMSE
  - R²
- Error analysis:
  - Distribution of prediction errors
  - Performance by player position

---

### 🎯 Deployment

- Built with **Gradio**
- Hosted on **Hugging Face Spaces**
- Real-time predictions
- Visual output on NFL field

---

## 🎮 App Features

- Interactive input panel for player tracking variables  
- Real-time ball landing prediction  
- Field visualization with:
  - Player position  
  - Predicted landing point  
  - Trajectory line  
- Preloaded example scenarios  

---

## 📁 Project Structure

```
nfl-big-data-bowl-2026/
│
├── app/                      # Gradio deployment app
│   └── app.py
│
├── models/                   # Deployment-ready models
│   └── final_deploy_xgboost/
│
├── notebooks/                # Full ML workflow
│   ├── 03_feature_engineering_v3.ipynb
│   ├── 04_model_training_v3.ipynb
│   ├── 04_model_training_final_deploy.ipynb
│   └── 05_model_evaluation_v3.ipynb
│
├── src/                      # Reusable pipeline code
│   ├── features.py
│   ├── deploy_features.py
│   └── config.py
│
├── assets/                   # Images / logo
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/nfl-big-data-bowl-2026.git
cd nfl-big-data-bowl-2026
python -m venv .venv
```

Activate environment:

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
cd app
python app.py
```

Open:

```
http://localhost:7860
```

---

## 📊 Results Summary

- XGBoost outperformed the baseline Random Forest  
- Improved prediction stability across player roles  
- Better modeling of nonlinear player motion dynamics  

---

## 🔮 Future Improvements

- Add prediction uncertainty (confidence intervals)
- Model multi-player interactions
- Sequence-based models (LSTM / Transformers)
- Real-time play simulation
- Integration with live tracking data

---

## 🧑‍💻 Author

**Jorge Montanez**  
Mechatronics Engineer | AI & Data Science  

- Machine Learning Systems  
- Data Science & Modeling  
- Real-world AI Deployment  

---

## 📜 License

This project is for educational and research purposes as part of the NFL Big Data Bowl.

---

## ⭐ Support

If you found this project interesting or useful:

👉 Give it a ⭐ on GitHub
