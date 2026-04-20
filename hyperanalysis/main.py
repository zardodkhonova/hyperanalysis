
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, \
    precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="Hypertension Predictor", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; font-weight: bold;}
.section-header {font-size: 1.5rem; color: #2e8b57; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #2e8b57; padding-bottom: 0.5rem;}
</style>""", unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">🏥 Hypertension Transition Predictor</h1>', unsafe_allow_html=True)

    # Initialize session state
    for key in ['data', 'processed_data', 'model', 'scaler', 'X_test', 'y_test', 'model_type', 'selected_features',
                'feature_selection', 'original_feature_names']:
        if key not in st.session_state:
            st.session_state[key] = None

    # Sidebar Navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Choose:", ["📁 Data Upload", "🔍 EDA", "🛠️ Preprocessing", "🤖 Training", "📈 Results",
                                            "🔮 Predictions", "📅 Time Series Predictions"])

    # Route to pages
    pages = {
        "📁 Data Upload": data_upload_page,
        "🔍 EDA": eda_page,
        "🛠️ Preprocessing": preprocessing_page,
        "🤖 Training": model_training_page,
        "📈 Results": results_page,
        "🔮 Predictions": prediction_page,
        "📅 Time Series Predictions": time_series_prediction_page
    }
    pages[page]()


def generate_sample_data():
    """Generate enhanced sample data with gender"""
    np.random.seed(42)
    n_participants, days_per_participant = 100, 20
    data = []

    for participant in range(1, n_participants + 1):
        # Random gender assignment
        gender = np.random.choice(['Male', 'Female'],
                                  p=[0.55, 0.45])  # Slightly more males (realistic for hypertension studies)

        # Gender-based BP baselines (males typically have slightly higher BP)
        if gender == 'Male':
            base_systolic = np.random.normal(145, 15)
            base_diastolic = np.random.normal(92, 8)
        else:
            base_systolic = np.random.normal(138, 15)
            base_diastolic = np.random.normal(88, 8)

        for day in range(days_per_participant):
            improvement_factor = np.random.choice([0, 1], p=[0.7, 0.3])

            if improvement_factor:
                systolic = base_systolic - day * np.random.normal(1.5, 0.5) + np.random.normal(0, 3)
                diastolic = base_diastolic - day * np.random.normal(0.8, 0.3) + np.random.normal(0, 2)
            else:
                systolic = base_systolic + np.random.normal(0, 5)
                diastolic = base_diastolic + np.random.normal(0, 3)

            systolic, diastolic = max(90, systolic), max(60, diastolic)

            # Determine threshold change
            threshold_change = 'to-nonhyper' if (
                        systolic < 130 and diastolic < 80 and improvement_factor and np.random.random() > 0.2) else np.random.choice(
                ['to-hyper', 'none'], p=[0.1, 0.9])

            data.append({
                'participant_id': participant,
                'gender': gender,
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day),
                'blood_pressure_systolic': systolic,
                'blood_pressure_diastolic': diastolic,
                'avg_heart_rate': np.random.normal(75, 8),
                'stress_level': np.random.randint(1, 11),
                'sleep_hours': np.random.normal(7, 1),
                'daily_steps': np.random.normal(8000, 2000),
                'activity_type': np.random.choice(['Running', 'Walking', 'Cycling', 'None']),
                'intensity': np.random.choice(['Low', 'Medium', 'High']),
                'threshold_change': threshold_change
            })

    return pd.DataFrame(data)


def data_upload_page():
    st.markdown('<div class="section-header" style="font-weight:bold; font-size:24px;">📂 Data Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success(f"✔️ Data uploaded successfully! Shape: {df.shape}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
    else:
        st.info("⬆️ Please upload a CSV file or generate sample data.")
        if st.button("⚙️ Generate Sample Data"):
            st.session_state.data = generate_sample_data()
            st.success("✔️ Sample data generated successfully!")
            st.experimental_rerun()
        return

    df = st.session_state.data

    # Basic info metrics with professional Unicode symbols
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Records", df.shape[0])
    with col2:
        st.metric("📋 Features", df.shape[1])
    with col3:
        to_nonhyper = df['threshold_change'].value_counts().get('to-nonhyper', 0)
        st.metric("🎯 To Non-Hypertensive", to_nonhyper)

    st.markdown("---")

    # Numerical summary for selected key columns
    key_numeric_cols = ['blood_pressure_systolic', 'blood_pressure_diastolic', 'avg_heart_rate', 'sleep_hours', 'daily_steps']
    available_cols = [col for col in key_numeric_cols if col in df.columns]

    st.subheader("📊 Numerical Summary Statistics")
    for col in available_cols:
        series = df[col].dropna()
        st.markdown(f"**{col.replace('_', ' ').title()}**")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mean", f"{series.mean():.2f}")
        col2.metric("Median", f"{series.median():.2f}")
        col3.metric("Min", f"{series.min():.2f}")
        col4.metric("Max", f"{series.max():.2f}")
        col5.metric("Std Dev", f"{series.std():.2f}")
        st.markdown("---")

    # Threshold change counts with clean labels and symbols
    if 'threshold_change' in df.columns:
        st.subheader("🎯 Threshold Change Counts")
        counts = df['threshold_change'].value_counts()
        for label, count in counts.items():
            st.write(f"• **{label.replace('-', ' ').title()}**: {count}")

def eda_page():
    st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return

    df = st.session_state.data.copy()

    # Check columns
    required_general = {'age', 'gender', 'threshold_change'}
    if required_general.issubset(df.columns):

        st.subheader("📊 Age Distribution by Hypertension State")
        fig = px.histogram(df, x='age', color='threshold_change',
                           nbins=30, barmode='overlay',
                           color_discrete_sequence=['#1f77b4', '#d62728', '#2ca02c'],
                           labels={'threshold_change': 'Hypertension Status'})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("👥 Gender vs Hypertension State")
        gender_dist = pd.crosstab(df['gender'], df['threshold_change'])
        fig = px.bar(gender_dist, barmode='group',
                     title="Gender vs Hypertension Transition",
                     labels={'value': 'Count', 'gender': 'Gender'},
                     color_discrete_sequence=['#1f77b4', '#d62728', '#2ca02c'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Skipping general analysis: missing one or more of `age`, `gender`, `threshold_change` columns.")

    # --- Participant-Specific BP Visualization ---
    st.markdown('<div class="section-header">🩺 Blood Pressure Trend by Participant</div>', unsafe_allow_html=True)

    required_cols = {'participant_id', 'date', 'blood_pressure_systolic', 'blood_pressure_diastolic'}
    if not required_cols.issubset(df.columns):
        st.error(f"❌ Missing columns: {required_cols - set(df.columns)}")
        return

    df['date'] = pd.to_datetime(df['date'])

    participant_ids = sorted(df['participant_id'].dropna().unique())
    selected_id = st.selectbox("👤 Select Participant ID", participant_ids)

    participant_data = df[df['participant_id'] == selected_id].sort_values("date")

    if participant_data.empty:
        st.warning("No data available for this participant.")
        return

    st.subheader(f"📈 Blood Pressure Over Time - Participant {selected_id}")
    fig = px.line(participant_data,
                  x='date',
                  y=['blood_pressure_systolic', 'blood_pressure_diastolic'],
                  labels={'value': 'Blood Pressure (mmHg)', 'variable': 'Type'},
                  title='Systolic & Diastolic Blood Pressure Over Time',
                  color_discrete_map={
                      'blood_pressure_systolic': '#1f77b4',
                      'blood_pressure_diastolic': '#d62728'
                  })
    fig.update_layout(legend_title_text='Blood Pressure Type')
    st.plotly_chart(fig, use_container_width=True)


def preprocess_data(df):
    """Streamlined preprocessing"""
    # Handle missing values
    if 'participant_id' in df.columns:
        df = df.sort_values(['participant_id', 'date']).groupby('participant_id').fillna(method='ffill').fillna(
            method='bfill')
    else:
        df = df.fillna(method='ffill').fillna(method='bfill')

    # Binary target
    df['target'] = (df['threshold_change'] == 'to-nonhyper').astype(int)

    # Core features
    core_features = ['blood_pressure_systolic', 'blood_pressure_diastolic', 'avg_heart_rate', 'stress_level',
                     'sleep_hours', 'daily_steps']
    available_features = [f for f in core_features if f in df.columns]

    # Feature engineering
    if 'participant_id' in df.columns and 'blood_pressure_systolic' in df.columns:
        df = df.sort_values(['participant_id', 'date'])
        df['bp_systolic_trend'] = df.groupby('participant_id')['blood_pressure_systolic'].diff()
        df['bp_diastolic_trend'] = df.groupby('participant_id')['blood_pressure_diastolic'].diff()
        available_features.extend(['bp_systolic_trend', 'bp_diastolic_trend'])

    if 'blood_pressure_systolic' in df.columns and 'blood_pressure_diastolic' in df.columns:
        df['bp_ratio'] = df['blood_pressure_systolic'] / df['blood_pressure_diastolic']
        available_features.append('bp_ratio')

    # Encode categoricals
    categorical_features = []
    for cat_col in ['activity_type', 'intensity', 'gender']:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            categorical_features.extend(dummies.columns.tolist())

    final_features = available_features + categorical_features + ['target']
    return df[final_features].dropna()


def preprocessing_page():
    st.markdown('<div class="section-header">🛠️ Preprocessing</div>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return

    if st.button("🚀 Process Data"):
        with st.spinner("Processing..."):
            processed_df = preprocess_data(st.session_state.data.copy())
            st.session_state.processed_data = processed_df
            st.success("✅ Preprocessing completed!")

            col1, col2 = st.columns(2)
            with col1: st.metric("Original Shape",
                                 f"{st.session_state.data.shape[0]} × {st.session_state.data.shape[1]}")
            with col2: st.metric("Processed Shape", f"{processed_df.shape[0]} × {processed_df.shape[1]}")

            feature_cols = [col for col in processed_df.columns if col != 'target']
            st.write(f"**Features:** {', '.join(feature_cols)}")
            st.dataframe(processed_df.head(), use_container_width=True)

            if 'target' in processed_df.columns:
                target_dist = processed_df['target'].value_counts()
                fig = px.bar(x=['No Transition', 'To Non-Hyper'], y=[target_dist.get(0, 0), target_dist.get(1, 0)])
                st.plotly_chart(fig, use_container_width=True)


def train_model(X_train, y_train, X_test, y_test, model_type):
    """Unified training function"""
    if model_type == "LSTM":
        X_train_lstm = X_train.reshape(-1, 1, X_train.shape[1])
        X_test_lstm = X_test.reshape(-1, 1, X_test.shape[1])

        model = Sequential([
            LSTM(32, input_shape=(1, X_train.shape[1])),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.AUC(name='auc')])
        history = model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, batch_size=32,
                            callbacks=[EarlyStopping(patience=7, restore_best_weights=True)], verbose=0)
        return model, history
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        return model, None


def model_training_page():
    st.markdown('<div class="section-header">🤖 Model Training</div>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("⚠️ Please preprocess data first!")
        return

    df = st.session_state.processed_data

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Model Type:", ["LSTM", "Random Forest"])
        feature_selection = st.checkbox("Feature Selection", value=True)
    with col2:
        use_smote = st.checkbox("SMOTE (Class Balance)", value=True)
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

    if st.button("🚀 Train Model"):
        with st.spinner("Training..."):
            X, y = df.drop('target', axis=1), df['target']

            # Feature selection
            if feature_selection:
                selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()]
            else:
                X_selected, selected_features = X.values, X.columns

            # SMOTE
            if use_smote:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_selected, y)
            else:
                X_resampled, y_resampled = X_selected, y

            # Split and scale
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size,
                                                                random_state=42, stratify=y_resampled)
            scaler = StandardScaler()
            X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

            # Train
            model, history = train_model(X_train_scaled, y_train, X_test_scaled, y_test, model_type)

            # Store results
            st.session_state.update({
                'model': model, 'scaler': scaler, 'X_test': X_test_scaled, 'y_test': y_test,
                'model_type': model_type, 'selected_features': selected_features,
                'feature_selection': feature_selection, 'original_feature_names': X.columns.tolist()
            })

            st.success("✅ Training completed!")

            if history and model_type == "LSTM":
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'AUC'))
                fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'), row=1, col=1)
                fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'), row=1, col=1)
                fig.add_trace(go.Scatter(y=history.history['auc'], name='Train AUC'), row=1, col=2)
                fig.add_trace(go.Scatter(y=history.history['val_auc'], name='Val AUC'), row=1, col=2)
                st.plotly_chart(fig, use_container_width=True)


def results_page():
    st.markdown('<div class="section-header">📈 Results</div>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first!")
        return

    model, X_test, y_test, model_type = st.session_state.model, st.session_state.X_test, st.session_state.y_test, st.session_state.model_type

    # Predictions
    if model_type == "LSTM":
        X_test_reshaped = X_test.reshape(-1, 1, X_test.shape[1])
        y_pred_prob = model.predict(X_test_reshaped, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred_prob, y_pred = model.predict_proba(X_test)[:, 1], model.predict(X_test)

    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_prob)
    }

    cols = st.columns(5)
    for i, (name, value) in enumerate(metrics.items()):
        cols[i].metric(name, f"{value:.3f}")

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fig = go.Figure([
            go.Scatter(x=fpr, y=tpr, name=f'Model (AUC = {metrics["AUC"]:.3f})'),
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random')
        ])
        fig.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
        st.plotly_chart(fig, use_container_width=True)

def prediction_page():
    st.markdown('<div class="section-header">🔮 Predictions</div>', unsafe_allow_html=True)

    # Ensure model and required components exist
    if "model" not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ Please train a model first!")
        return
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("📂 Please upload or generate data to determine input ranges.")
        return
    if "scaler" not in st.session_state:
        st.warning("⚠️ Scaler not found in session state.")
        return
    if "original_feature_names" not in st.session_state:
        st.warning("⚠️ Original feature names not found.")
        return

    df = st.session_state.data

    def get_range(col, default_min, default_max, default_median):
        if col in df.columns:
            return float(df[col].min()), float(df[col].max()), float(df[col].median())
        else:
            return default_min, default_max, default_median

    sys_min, sys_max, sys_median = get_range('blood_pressure_systolic', 90, 200, 140)
    dia_min, dia_max, dia_median = get_range('blood_pressure_diastolic', 50, 120, 90)
    hr_min, hr_max, hr_median = get_range('avg_heart_rate', 40, 200, 75)
    sleep_min, sleep_max, sleep_median = get_range('sleep_hours', 3.0, 12.0, 7.0)
    steps_min, steps_max, steps_median = get_range('daily_steps', 0, 30000, 8000)

    # --- Input Form ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            systolic_bp = st.slider("Systolic BP (mmHg)", int(sys_min), int(sys_max), int(sys_median))
            diastolic_bp = st.slider("Diastolic BP (mmHg)", int(dia_min), int(dia_max), int(dia_median))
            avg_hr = st.slider("Heart Rate (bpm)", int(hr_min), int(hr_max), int(hr_median))
        with col2:
            stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
            sleep_hours = st.slider("Sleep Hours", float(sleep_min), float(sleep_max), float(sleep_median), step=0.5)
            daily_steps = st.slider("Daily Steps", int(steps_min), int(steps_max), int(steps_median), step=100)

        predict_button = st.form_submit_button("🔮 Predict")

    # --- Prediction Logic ---
    if predict_button:
        try:
            original_features = st.session_state.original_feature_names
            feature_dict = {f: 0.0 for f in original_features}

            inputs = {
                'blood_pressure_systolic': systolic_bp,
                'blood_pressure_diastolic': diastolic_bp,
                'avg_heart_rate': avg_hr,
                'stress_level': stress_level,
                'sleep_hours': sleep_hours,
                'daily_steps': daily_steps
            }

            # Populate feature dict
            for feature, value in inputs.items():
                if feature in feature_dict:
                    feature_dict[feature] = value

            if 'bp_ratio' in feature_dict:
                feature_dict['bp_ratio'] = systolic_bp / diastolic_bp

            # Prepare features
            features = np.array([feature_dict[f] for f in original_features]).reshape(1, -1)

            # Apply feature selection
            if st.session_state.feature_selection:
                selected_features = st.session_state.selected_features
                selected_indices = [original_features.index(f) for f in selected_features if f in original_features]
                features = features[:, selected_indices]

            # Scale and predict
            scaler = st.session_state.scaler
            model = st.session_state.model

            features_scaled = scaler.transform(features)

            if st.session_state.model_type == "LSTM":
                features_reshaped = features_scaled.reshape(-1, 1, features_scaled.shape[1])
                prediction_prob = float(model.predict(features_reshaped, verbose=0)[0][0])
            else:
                prediction_prob = float(model.predict_proba(features_scaled)[0][1])

            # --- Result Display ---
            col1, col2 = st.columns(2)
            with col1:
                result = "🟢 Likely to Transition" if prediction_prob > 0.5 else "🔴 Unlikely to Transition"
                st.metric("Prediction", result, f"Confidence: {prediction_prob:.1%}")
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prediction_prob * 100,
                    title={'text': "Transition Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "red"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            # --- Recommendations ---
            st.subheader("💡 Recommendations")
            if prediction_prob > 0.7:
                st.success("**Excellent Progress!** Keep going:")
                recs = [
                    "✔️ Maintain current lifestyle habits",
                    "🩺 Regular medical checkups",
                    "📈 Keep tracking your blood pressure"
                ]
            elif prediction_prob > 0.4:
                st.warning("**Moderate chance – Improve lifestyle:**")
                recs = [
                    "🥗 Adopt DASH diet (low sodium)",
                    "🏃‍♀️ Exercise 150+ minutes/week",
                    "🧘‍♂️ Manage stress actively"
                ]
            else:
                st.error("**High Risk – Immediate action needed!**")
                recs = [
                    "⚠️ Consult a healthcare provider",
                    "💊 Possible medication adjustment",
                    "🍎 Strict diet and exercise plan"
                ]

            for rec in recs:
                st.write(f"- {rec}")

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))


def get_participant_predictions(participant_id, start_date, num_days):
    """Generate time series predictions for a participant"""
    if st.session_state.data is None or st.session_state.model is None:
        return None

    # Get participant's historical data
    df = st.session_state.data.copy()
    participant_data = df[
        df['participant_id'] == participant_id].copy() if 'participant_id' in df.columns else df.copy()

    if len(participant_data) == 0:
        return None

    # Get last known values for the participant
    last_row = participant_data.iloc[-1]

    predictions = []
    current_date = pd.to_datetime(start_date)

    # Use last known values as baseline
    baseline_values = {
        'blood_pressure_systolic': last_row.get('blood_pressure_systolic', 140),
        'blood_pressure_diastolic': last_row.get('blood_pressure_diastolic', 90),
        'avg_heart_rate': last_row.get('avg_heart_rate', 75),
        'stress_level': last_row.get('stress_level', 5),
        'sleep_hours': last_row.get('sleep_hours', 7),
        'daily_steps': last_row.get('daily_steps', 8000),
        'gender': last_row.get('gender', 'Male'),
        'activity_type': last_row.get('activity_type', 'Walking'),
        'intensity': last_row.get('intensity', 'Medium')
    }

    for day in range(num_days):
        # Add some realistic variation
        daily_variation = {
            'blood_pressure_systolic': baseline_values['blood_pressure_systolic'] + np.random.normal(0, 3),
            'blood_pressure_diastolic': baseline_values['blood_pressure_diastolic'] + np.random.normal(0, 2),
            'avg_heart_rate': baseline_values['avg_heart_rate'] + np.random.normal(0, 5),
            'stress_level': max(1, min(10, baseline_values['stress_level'] + np.random.randint(-1, 2))),
            'sleep_hours': max(4, min(12, baseline_values['sleep_hours'] + np.random.normal(0, 0.5))),
            'daily_steps': max(0, baseline_values['daily_steps'] + np.random.normal(0, 1000))
        }

        # Create feature vector
        try:
            original_features = st.session_state.original_feature_names
            feature_dict = {f: 0.0 for f in original_features}

            # Fill in the values
            for feature, value in daily_variation.items():
                if feature in feature_dict:
                    feature_dict[feature] = value

            # Add engineered features
            if 'bp_ratio' in feature_dict:
                feature_dict['bp_ratio'] = daily_variation['blood_pressure_systolic'] / daily_variation[
                    'blood_pressure_diastolic']

            # Handle categorical features
            if 'gender_Male' in feature_dict:
                feature_dict['gender_Male'] = 1 if baseline_values['gender'] == 'Male' else 0
            if 'activity_type_Running' in feature_dict:
                feature_dict['activity_type_Running'] = 1 if baseline_values['activity_type'] == 'Running' else 0
            if 'activity_type_Walking' in feature_dict:
                feature_dict['activity_type_Walking'] = 1 if baseline_values['activity_type'] == 'Walking' else 0
            if 'activity_type_Cycling' in feature_dict:
                feature_dict['activity_type_Cycling'] = 1 if baseline_values['activity_type'] == 'Cycling' else 0
            if 'intensity_Medium' in feature_dict:
                feature_dict['intensity_Medium'] = 1 if baseline_values['intensity'] == 'Medium' else 0
            if 'intensity_High' in feature_dict:
                feature_dict['intensity_High'] = 1 if baseline_values['intensity'] == 'High' else 0

            features = np.array([feature_dict[f] for f in original_features]).reshape(1, -1)

            # Apply feature selection if used
            if st.session_state.feature_selection:
                selected_features = st.session_state.selected_features
                selected_indices = [original_features.index(f) for f in selected_features if f in original_features]
                features = features[:, selected_indices]

            # Scale and predict
            features_scaled = st.session_state.scaler.transform(features)

            if st.session_state.model_type == "LSTM":
                features_reshaped = features_scaled.reshape(-1, 1, features_scaled.shape[1])
                prediction_prob = float(st.session_state.model.predict(features_reshaped, verbose=0)[0][0])
            else:
                prediction_prob = float(st.session_state.model.predict_proba(features_scaled)[0][1])

            predictions.append({
                'date': current_date,
                'systolic_bp': daily_variation['blood_pressure_systolic'],
                'diastolic_bp': daily_variation['blood_pressure_diastolic'],
                'heart_rate': daily_variation['avg_heart_rate'],
                'transition_probability': prediction_prob,
                'transition_likely': prediction_prob > 0.5
            })

        except Exception as e:
            st.error(f"Error in prediction for day {day}: {str(e)}")
            break

        current_date += timedelta(days=1)

    return pd.DataFrame(predictions)


def get_participant_transition_probability(participant_id, df):
    """Calculate average transition probability for a participant"""
    try:
        if st.session_state.model is None or st.session_state.scaler is None:
            return 0.0

        participant_data = df[df['participant_id'] == participant_id] if 'participant_id' in df.columns else df

        if len(participant_data) == 0:
            return 0.0

        # Get last known values for baseline
        last_row = participant_data.iloc[-1]
        baseline_values = {
            'blood_pressure_systolic': last_row.get('blood_pressure_systolic', 140),
            'blood_pressure_diastolic': last_row.get('blood_pressure_diastolic', 90),
            'avg_heart_rate': last_row.get('avg_heart_rate', 75),
            'stress_level': last_row.get('stress_level', 5),
            'sleep_hours': last_row.get('sleep_hours', 7),
            'daily_steps': last_row.get('daily_steps', 8000),
            'gender': last_row.get('gender', 'Male'),
            'activity_type': last_row.get('activity_type', 'Walking'),
            'intensity': last_row.get('intensity', 'Medium')
        }

        # Create feature vector using baseline values
        original_features = st.session_state.original_feature_names
        feature_dict = {f: 0.0 for f in original_features}

        # Fill in the baseline values
        for feature, value in baseline_values.items():
            if feature in feature_dict:
                feature_dict[feature] = value

        # Add engineered features
        if 'bp_ratio' in feature_dict:
            feature_dict['bp_ratio'] = baseline_values['blood_pressure_systolic'] / baseline_values[
                'blood_pressure_diastolic']

        # Handle categorical features
        if 'gender_Male' in feature_dict:
            feature_dict['gender_Male'] = 1 if baseline_values['gender'] == 'Male' else 0
        if 'activity_type_Running' in feature_dict:
            feature_dict['activity_type_Running'] = 1 if baseline_values['activity_type'] == 'Running' else 0
        if 'activity_type_Walking' in feature_dict:
            feature_dict['activity_type_Walking'] = 1 if baseline_values['activity_type'] == 'Walking' else 0
        if 'activity_type_Cycling' in feature_dict:
            feature_dict['activity_type_Cycling'] = 1 if baseline_values['activity_type'] == 'Cycling' else 0
        if 'intensity_Medium' in feature_dict:
            feature_dict['intensity_Medium'] = 1 if baseline_values['intensity'] == 'Medium' else 0
        if 'intensity_High' in feature_dict:
            feature_dict['intensity_High'] = 1 if baseline_values['intensity'] == 'High' else 0

        features = np.array([feature_dict[f] for f in original_features]).reshape(1, -1)

        # Apply feature selection if used
        if st.session_state.feature_selection:
            selected_features = st.session_state.selected_features
            selected_indices = [original_features.index(f) for f in selected_features if f in original_features]
            features = features[:, selected_indices]

        # Scale and predict
        features_scaled = st.session_state.scaler.transform(features)

        if st.session_state.model_type == "LSTM":
            features_reshaped = features_scaled.reshape(-1, 1, features_scaled.shape[1])
            prediction_prob = float(st.session_state.model.predict(features_reshaped, verbose=0)[0][0])
        else:
            prediction_prob = float(st.session_state.model.predict_proba(features_scaled)[0][1])

        return prediction_prob

    except Exception as e:
        return 0.0


def filter_participants_by_bp_and_probability(df, min_probability=0.65):
    """Filter participants based on blood pressure categories and transition probability:
    - BP Categories: Optimal, Normal, High Normal
    - Transition Probability: >= min_probability (default 65%)
    """
    if 'participant_id' not in df.columns:
        return [1]  # Default for sample data without participant_id

    # Required columns for blood pressure filtering
    required_cols = ['blood_pressure_systolic', 'blood_pressure_diastolic']
    if not all(col in df.columns for col in required_cols):
        st.warning("Blood pressure data not found. Showing all participants.")
        return sorted(df['participant_id'].unique())

    valid_participants = []
    participant_probabilities = {}

    for participant_id in df['participant_id'].unique():
        participant_data = df[df['participant_id'] == participant_id]

        # Check if participant has any readings in the desired BP categories
        has_valid_bp = False

        for _, row in participant_data.iterrows():
            systolic = row['blood_pressure_systolic']
            diastolic = row['blood_pressure_diastolic']

            # Skip if BP values are missing or invalid
            if pd.isna(systolic) or pd.isna(diastolic):
                continue

            # Check BP categories
            # Optimal: < 120 and < 80
            optimal = (systolic < 120) and (diastolic < 80)

            # Normal: 120-129 and/or < 80
            normal = ((120 <= systolic <= 129) and (diastolic < 80)) or \
                     ((systolic < 120) and (diastolic < 80))

            # High Normal: 130-139 and/or 80-84
            high_normal = ((130 <= systolic <= 139) and (diastolic <= 84)) or \
                          ((systolic <= 139) and (80 <= diastolic <= 84))

            if optimal or normal or high_normal:
                has_valid_bp = True
                break

        # If BP is valid, check transition probability
        if has_valid_bp:
            transition_prob = get_participant_transition_probability(participant_id, df)
            participant_probabilities[participant_id] = transition_prob

            if transition_prob >= min_probability:
                valid_participants.append(participant_id)

    # Store probabilities in session state for display
    st.session_state.participant_probabilities = participant_probabilities

    return sorted(valid_participants)


def time_series_prediction_page():
    st.markdown('<div class="section-header">📅 Time Series Predictions</div>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first!")
        return

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return

    # Add probability threshold selector
    st.subheader("🎯 Filter Settings")
    col1, col2 = st.columns(2)

    with col1:
        min_probability = st.slider(
            "Minimum Transition Probability (%)",
            min_value=50,
            max_value=95,
            value=65,
            step=5,
            help="Only show participants with transition probability above this threshold"
        ) / 100.0

    with col2:
        show_all_participants = st.checkbox(
            "Show all participants (ignore probability filter)",
            value=False,
            help="Check this to see all participants regardless of probability"
        )

    # Get available participants filtered by BP categories and probability
    df = st.session_state.data

    if show_all_participants:
        # Use original BP-only filter
        participant_ids = []
        if 'participant_id' in df.columns:
            for participant_id in df['participant_id'].unique():
                participant_data = df[df['participant_id'] == participant_id]
                has_valid_bp = False

                for _, row in participant_data.iterrows():
                    systolic = row.get('blood_pressure_systolic', 0)
                    diastolic = row.get('blood_pressure_diastolic', 0)

                    if pd.isna(systolic) or pd.isna(diastolic):
                        continue

                    optimal = (systolic < 120) and (diastolic < 80)
                    normal = ((120 <= systolic <= 129) and (diastolic < 80)) or ((systolic < 120) and (diastolic < 80))
                    high_normal = ((130 <= systolic <= 139) and (diastolic <= 84)) or (
                                (systolic <= 139) and (80 <= diastolic <= 84))

                    if optimal or normal or high_normal:
                        has_valid_bp = True
                        break

                if has_valid_bp:
                    participant_ids.append(participant_id)
        else:
            participant_ids = [1]
    else:
        participant_ids = filter_participants_by_bp_and_probability(df, min_probability)

    if not participant_ids:
        st.error("❌ No participants found matching the criteria:")
        st.write("• **Blood Pressure Categories**: Optimal, Normal, or High Normal")
        st.write(f"• **Minimum Transition Probability**: {min_probability:.0%}")
        st.write("")
        st.write("**Blood Pressure Categories:**")
        st.write("• **Optimal**: Systolic < 120 and Diastolic < 80")
        st.write("• **Normal**: Systolic 120-129 and/or Diastolic < 80")
        st.write("• **High Normal**: Systolic 130-139 and/or Diastolic 80-84")
        return

    # Display filtering information
    if not show_all_participants:
        st.success(f"🎯 Found {len(participant_ids)} participants with transition probability ≥ {min_probability:.0%}")

        # Show participant probabilities if available
        if hasattr(st.session_state, 'participant_probabilities'):
            st.subheader("📊 Participant Transition Probabilities")
            prob_data = []
            for pid in participant_ids:
                if pid in st.session_state.participant_probabilities:
                    prob_data.append({
                        'Participant ID': pid,
                        'Transition Probability': f"{st.session_state.participant_probabilities[pid]:.1%}"
                    })

            if prob_data:
                prob_df = pd.DataFrame(prob_data)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"📊 Showing {len(participant_ids)} participants (all with valid BP categories)")

    # Display BP category information
    st.write("**Included Blood Pressure Categories:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**Optimal**: < 120 / < 80")
    with col2:
        st.info("**Normal**: 120-129 / < 80")
    with col3:
        st.warning("**High Normal**: 130-139 / 80-84")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_participant = st.selectbox("Select Participant:", participant_ids)

    with col2:
        start_date = st.date_input("Start Date:", value=datetime.now().date())

    with col3:
        num_days = st.slider("Number of Days:", 1, 30, 14)

    if st.button("🔮 Generate Time Series Predictions"):
        with st.spinner("Generating predictions..."):
            predictions_df = get_participant_predictions(selected_participant, start_date, num_days)

            if predictions_df is not None and len(predictions_df) > 0:
                st.success(f"✅ Generated {len(predictions_df)} day predictions!")

                # Time series visualization
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Blood Pressure Trends', 'Heart Rate Trend', 'Transition Probability'),
                    vertical_spacing=0.1
                )

                # Blood pressure plot
                fig.add_trace(
                    go.Scatter(x=predictions_df['date'], y=predictions_df['systolic_bp'],
                               name='Systolic BP', line=dict(color='red')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=predictions_df['date'], y=predictions_df['diastolic_bp'],
                               name='Diastolic BP', line=dict(color='blue')),
                    row=1, col=1
                )

                # Heart rate plot
                fig.add_trace(
                    go.Scatter(x=predictions_df['date'], y=predictions_df['heart_rate'],
                               name='Heart Rate', line=dict(color='green')),
                    row=2, col=1
                )

                # Transition probability plot
                fig.add_trace(
                    go.Scatter(x=predictions_df['date'], y=predictions_df['transition_probability'],
                               name='Transition Probability', line=dict(color='purple')),
                    row=3, col=1
                )

                # Add threshold line for transition probability
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=3, col=1)

                fig.update_layout(height=800, title_text=f"Participant {selected_participant} - Health Predictions")
                fig.update_xaxes(title_text="Date", row=3, col=1)
                fig.update_yaxes(title_text="BP (mmHg)", row=1, col=1)
                fig.update_yaxes(title_text="BPM", row=2, col=1)
                fig.update_yaxes(title_text="Probability", row=3, col=1)

                st.plotly_chart(fig, use_container_width=True)

                # Summary statistics
                st.subheader("📊 Prediction Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    avg_transition_prob = predictions_df['transition_probability'].mean()
                    st.metric("Avg Transition Probability", f"{avg_transition_prob:.1%}")

                with col2:
                    likely_days = (predictions_df['transition_probability'] > 0.5).sum()
                    st.metric("Days with High Probability", f"{likely_days}/{len(predictions_df)}")

                with col3:
                    avg_systolic = predictions_df['systolic_bp'].mean()
                    st.metric("Avg Systolic BP", f"{avg_systolic:.0f} mmHg")

                with col4:
                    avg_diastolic = predictions_df['diastolic_bp'].mean()
                    st.metric("Avg Diastolic BP", f"{avg_diastolic:.0f} mmHg")

                # Daily breakdown table
                st.subheader("📋 Daily Breakdown")

                # Format the predictions for display
                display_df = predictions_df.copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                display_df['systolic_bp'] = display_df['systolic_bp'].round(0).astype(int)
                display_df['diastolic_bp'] = display_df['diastolic_bp'].round(0).astype(int)
                display_df['heart_rate'] = display_df['heart_rate'].round(0).astype(int)
                display_df['transition_probability'] = display_df['transition_probability'].apply(lambda x: f"{x:.1%}")
                display_df['transition_likely'] = display_df['transition_likely'].apply(
                    lambda x: "✅ Yes" if x else "❌ No")

                # Rename columns for better display
                display_df = display_df.rename(columns={
                    'date': 'Date',
                    'systolic_bp': 'Systolic BP',
                    'diastolic_bp': 'Diastolic BP',
                    'heart_rate': 'Heart Rate',
                    'transition_probability': 'Transition Prob',
                    'transition_likely': 'Likely Transition'
                })

                st.dataframe(display_df, use_container_width=True)

                # Download predictions
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"participant_{selected_participant}_predictions.csv",
                    mime="text/csv"
                )

                # Health recommendations based on predictions
                st.subheader("💡 Health Recommendations")

                high_prob_days = (predictions_df['transition_probability'] > 0.7).sum()
                moderate_prob_days = ((predictions_df['transition_probability'] > 0.4) &
                                      (predictions_df['transition_probability'] <= 0.7)).sum()
                low_prob_days = (predictions_df['transition_probability'] <= 0.4).sum()

                if high_prob_days > len(predictions_df) * 0.6:
                    st.success("🎉 **Excellent outlook!** Most days show high transition probability.")
                    st.write("**Recommendations:**")
                    st.write("• Continue current lifestyle modifications")
                    st.write("• Regular monitoring and medical check-ups")
                    st.write("• Maintain medication adherence if prescribed")

                elif moderate_prob_days > len(predictions_df) * 0.5:
                    st.warning("⚠️ **Moderate outlook** - Room for improvement")
                    st.write("**Recommendations:**")
                    st.write("• Focus on consistent exercise routine")
                    st.write("• Optimize sleep schedule (7-9 hours)")
                    st.write("• Monitor sodium intake more closely")
                    st.write("• Consider stress management techniques")

                else:
                    st.error("🚨 **Requires attention** - Low transition probability")
                    st.write("**Recommendations:**")
                    st.write("• Consult healthcare provider for treatment review")
                    st.write("• Consider intensive lifestyle intervention")
                    st.write("• Daily BP monitoring recommended")
                    st.write("• Medication adjustment may be needed")

            else:
                st.error("❌ Could not generate predictions. Please check the data and model.")


if __name__ == "__main__":
    main()