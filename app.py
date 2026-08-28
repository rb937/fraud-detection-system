import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(page_title="Fraud Detection System", layout="wide")

def load_assets():
    model = joblib.load('model/fraud_model.pkl')
    le_cat = joblib.load('model/category_encoder.pkl')
    features = joblib.load('model/features.pkl')
    return model, le_cat, features

try:
    model, le_cat, feature_names = load_assets()
except:
    st.error("Model files not found! Please run 'train_model.ipynb' first to generate the model.")
    st.stop()

st.title("Fraud Detection System")
st.markdown("### AI-Driven Transaction Analysis")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Transaction Details")
    st.write("Simulate a transaction to see the risk score.")
    
    amount_inr = st.slider("Transaction Amount (₹)", 0, 600000, 12000, step=1000)
    amount_usd = round(amount_inr / 95.0, 2)
    st.caption(f"≈ ${amount_usd:,.2f} ({amount_inr:,} INR)")
    
    category = st.selectbox("Category", le_cat.classes_)
    gender = st.radio("Cardholder Gender", ["M", "F"])
    age = st.slider("Customer Age", 18, 90, 30)
    distance = st.slider("Distance from Home (km)", 0.0, 1000.0, 5.0)
    hour = st.slider("Hour of Day (24h)", 0, 23, 14)
    
    gender_enc = 0 if gender == "M" else 1
    category_enc = le_cat.transform([category])[0]
    
    input_data = pd.DataFrame({
        'category': [category_enc],
        'amt': [amount_usd],
        'gender': [gender_enc],
        'age': [age],
        'distance_km': [distance],
        'hour': [hour]
    })

with col2:
    st.header("Risk Analysis")
    
    if st.button("Analyze Transaction", use_container_width=True):
        probability = model.predict_proba(input_data)[0][1]
        
        st.subheader("Result:")
        
        gauge_color = "green"
        if probability > 0.7:
            gauge_color = "red"
            st.error(f"🚨 FRAUD DETECTED (Confidence: {probability:.1%})")
            st.warning("Reason: High likelihood of anomaly detected based on Amount and Distance.")
        elif probability > 0.3:
            gauge_color = "orange"
            st.warning(f"SUSPICIOUS ACTIVITY (Risk: {probability:.1%})")
        else:
            st.success(f"LEGITIMATE TRANSACTION (Safe: {1-probability:.1%})")
            
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            title = {'text': "Fraud Probability (%)"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}]}))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Transaction Geography")
    
    home_lat, home_lon = 28.6304, 77.2177
    
    offset = distance / 111.0
    merch_lat, merch_lon = home_lat + offset, home_lon + offset
    
    home_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame({'lat': [home_lat], 'lon': [home_lon]}),
        get_position='[lon, lat]',
        get_color='[0, 255, 0, 200]',
        get_radius=750,
        pickable=True,
    )
    
    radius_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame({'lat': [home_lat], 'lon': [home_lon]}),
        get_position='[lon, lat]',
        get_color='[0, 100, 255, 30]',
        get_radius=distance * 1000,
        get_line_color=[0, 100, 255],
        stroked=True,
        filled=True,
    )

    view_state = pdk.ViewState(
        latitude=home_lat,
        longitude=home_lon,
        zoom=7 if distance > 50 else 10,
        pitch=0,
    )
    
    r = pdk.Deck(
        layers=[radius_layer, home_layer],
        initial_view_state=view_state,
        tooltip={"text": "Location"}
    )
    
    st.pydeck_chart(r)
    st.caption(f"Visualizing a {distance} km radius around New Delhi.")