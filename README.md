HyperAnalysis — Hypertension Transition Predictor

A clinical decision-support web application that predicts whether a hypertensive patient is likely to normalize their blood pressure, using longitudinal biometric and lifestyle data combined with machine learning.

📌 Overview
HyperAnalysis is a multi-page Streamlit application designed for researchers and clinicians working with hypertensive patient data. It provides an end-to-end pipeline — from raw CSV ingestion to day-by-day health forecasting — with interactive visualizations and personalized health recommendations.
The core prediction task is binary classification: will a patient transition from hypertensive to non-hypertensive status based on their current biometric readings and lifestyle habits?

✨ Features
📁 Data Upload

Upload your own CSV dataset or generate a synthetic dataset (100 participants × 20 days)
Automatic summary statistics and class distribution display

🔍 Exploratory Data Analysis (EDA)

Age distribution by hypertension status
Gender vs. hypertension transition breakdown
Per-participant blood pressure trends over time (systolic & diastolic)

🛠️ Preprocessing

Forward/backward fill for missing values (grouped by participant)
Feature engineering: BP trend (day-over-day delta), BP ratio (systolic/diastolic)
One-hot encoding of categorical features (activity type, intensity, gender)
Binary target creation (to-nonhyper → 1, otherwise → 0)

🤖 Model Training

LSTM (Long Short-Term Memory) neural network via TensorFlow/Keras
Random Forest classifier via Scikit-learn
Optional SMOTE oversampling to handle class imbalance
Optional automatic feature selection using Random Forest importance
Configurable train/test split

📈 Results & Evaluation

Accuracy, Precision, Recall, F1-Score, AUC-ROC
Confusion Matrix heatmap
ROC Curve visualization

🔮 Single Prediction

Slider-based input form for manual patient data entry
Gauge chart showing transition probability
Risk-level health recommendations

📅 Time Series Forecasting

Day-by-day forecast of systolic BP, diastolic BP, heart rate, and transition probability
Up to 30-day lookahead per participant
Downloadable prediction CSV
Automated health recommendations based on forecast outlook


🧰 Tech Stack
CategoryLibraryWeb FrameworkStreamlitData ProcessingPandas, NumPyMachine LearningScikit-learnDeep LearningTensorFlow / Keras (LSTM)Class Balancingimbalanced-learn (SMOTE)VisualizationPlotly Express, Plotly Graph ObjectsLanguagePython 3.9+

🚀 Getting Started
1. Clone the repository
bashgit clone https://github.com/zardodkhonova/hyperanalysis.git
cd hyperanalysis
2. Install dependencies
bashpip install -r requirements.txt
3. Run the app
bashstreamlit run main.py
The app will open in your browser at http://localhost:8501.

📋 Requirements
Create a requirements.txt with the following:
streamlit
pandas
numpy
plotly
scikit-learn
imbalanced-learn
tensorflow



📄 License
This project is licensed under the MIT License.