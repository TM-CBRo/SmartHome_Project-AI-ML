import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib
from fuzzy_system import get_fuzzy_system

# Configurare Pagină
st.set_page_config(page_title="Smart Home AI by TIM", layout="wide")

# Încărcarea Modelului ML
@st.cache_resource
def load_model():
    return joblib.load('model_clima.pkl')

try:
    model = load_model()
except:
    st.error("Nu am găsit fișierul 'model_clima.pkl'. Rulează întâi ml_model.py!")
    st.stop()

st.title("Sistem avansat de control climatic powerd by AndreX ")
st.markdown("""
Acest sistem foloseste **Machine Learning** pentru a estima temperatura viitoare si **Inteligenta Artificiala (logica fuzzy)** pentru a lua decizii.
""")

col1, col2 = st.columns([1, 2])

# Simulare senzori (input)
with col1:
    st.subheader("Panou de Control")
    st.write("Simulare date senzori:")
    
    current_hour = st.slider("Ora Curentă", 0, 23, 14)
    
    
    current_temp = st.number_input("Temperatura Curentă (°C) de afara", value=10.0, step=0.1, format="%.2f")
    humidity = st.slider("Umiditate (%)", 0, 100, 60)
    pressure = st.number_input("Presiune Atmosferică (hPa)", value=1013.0, step=1.0, format="%.1f")
    
    st.markdown("---")
    target_temp = st.number_input("Temperatura Dorită în Casă", value=22.0, step=0.5, format="%.2f")

# Zona procesare (ML + AI) 
with col2:
    # predictie ML
    input_data = pd.DataFrame({
        'Ora': [current_hour],
        'Temperatura_Curenta': [current_temp],
        'Umiditate': [humidity],
        'Presiune': [pressure]
    })
    
    # predicția
    predicted_raw = model.predict(input_data)[0]
    
    # Rotunjim matematic variabila 
    predicted_temp = round(predicted_raw, 2)
    
    st.subheader("Modul Machine Learning (Predicție)")
    
    # Calculare delta separat și aplicăm formatare STRICTĂ pe toate afișările
    delta_val = predicted_temp - current_temp
    
    m1, m2, m3 = st.columns(3)
    
    # Aici forțăm afișarea cu 2 zecimale
    m1.metric("Acum", f"{current_temp:.2f} °C")
    m2.metric("Peste 1 oră (Predicție)", f"{predicted_temp:.2f} °C", delta=f"{delta_val:.2f} °C")
    
    # 2. Decizia FUZZY
    st.subheader("Modul Inteligență Artificială (Decizie)")
    
    temp_diff = target_temp - predicted_temp
    
    # Rulare sistemul Fuzzy
    sim = get_fuzzy_system()
    sim.input['temp_diff'] = temp_diff
    sim.input['humidity'] = humidity
    sim.compute()
    
    power_output = sim.output['power']
    
    # Formatare strictă și la mesajele text
    st.info(f"Diferența calculată: {temp_diff:.2f} °C (bazată pe viitor)")
    
    if power_output > 0:
        # Formatare la procentul de putere
        st.success(f"CENTRALA PORNEȘTE: Putere {power_output:.2f}%")
        st.progress(int(power_output))
    elif power_output < 0:
        st.info(f"AER CONDIȚIONAT: Putere {abs(power_output):.2f}%")
        st.progress(int(abs(power_output)))
    else:
        st.warning("Sistem în Standby")