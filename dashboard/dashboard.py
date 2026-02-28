import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
import plotly.express as px

# 1. Load Tools & Data
model = joblib.load('fraud_model_final.pkl')

# Load the test data you saved during training
@st.cache_data
def load_data():
    return pd.read_csv('test_data.csv')

try:
    df_test = load_data()
except:
    st.error("test_data.csv not found! Please save your X_test/y_test to a CSV first.")
    st.stop()

st.set_page_config(page_title="FraudShield AI", page_icon="🚨", layout="wide")
st.title("🚨 FraudShield: Real-Time Detection System")

# 2. Initialize Session State
if 'money_saved' not in st.session_state:
    st.session_state.money_saved = 0.0
if 'frauds_caught' not in st.session_state:
    st.session_state.frauds_caught = 0
if 'log_history' not in st.session_state:
    st.session_state.log_history = []

# 3. UI Placeholders
m1, m2, m3 = st.columns(3)
status_p = m1.empty()
saved_p = m2.empty()
caught_p = m3.empty()

st.subheader("Live Transaction Stream")
table_p = st.empty()

# 4. Monitoring Logic
# Add toggle for Start/Stop
run_monitoring = st.sidebar.toggle('Start Monitoring')

# Add chart placeholder at the top of the dashboard
chart_p = st.empty()

if run_monitoring:
    while True:
        # --- Real Data Sampling ---
        # Pick a random row from our real test set
        sample = df_test.sample(1)

        # Prepare features (Everything except 'Class')
        features_df = sample.drop('Class', axis=1)
        actual_label = sample['Class'].values[0]

        # Prediction - Pull amount and probability
        amount = float(sample['scaled_amount'].values[0])
        prob = float(model.predict_proba(features_df)[0][1])

        # Sliding Threshold logic
        threshold = 0.25 if amount > 5.0 else 0.55
        is_fraud = prob >= threshold
        status = "🚨 BLOCK" if is_fraud else "✅ ALLOW"

        # --- Metrics logic ---
        if is_fraud and actual_label == 1:
            st.session_state.money_saved += abs(amount)
            st.session_state.frauds_caught += 1

        # History Logging
        new_entry = {
            "Time": time.strftime("%H:%M:%S"),
            "Amount": round(amount, 2),
            "Prob": round(prob, 4),
            "Action": status,
            "Real Class": "Fraud" if actual_label == 1 else "Normal"
        }
        st.session_state.log_history.insert(0, new_entry)


        # --- Update Probability Trend Chart ---
        df_history = pd.DataFrame(st.session_state.log_history).head(50)
        if not df_history.empty:
            fig = px.line(df_history, x="Time", y="Prob",
                          title="Live Fraud Probability Trend",
                          range_y=[0, 1], line_shape="spline", markers=True)

            # Add a red line for threshold_used
            fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                          annotation_text=f"Current Threshold ({threshold})")

            chart_p.plotly_chart(fig, width='stretch')


        # --- Refresh UI Components ---
        status_p.metric("System Status", "LIVE", delta="Active")
        saved_p.metric("Money Saved (Scaled)", f"{st.session_state.money_saved:.2f}")
        caught_p.metric("Frauds Blocked", st.session_state.frauds_caught)

        table_p.table(pd.DataFrame(st.session_state.log_history).head(10))

        # Pause to prevent system overload
        time.sleep(0.5)
else:
    status_p.metric("System Status", "OFFLINE",delta="Stopped", delta_color="inverse")
    st.info("Toggle 'Start Monitoring' in the sidebar to begin the live stream.")
