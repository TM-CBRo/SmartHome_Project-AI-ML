import pandas as pd
import numpy as np

print("Procesare fisier weatherHistory.csv")

# Citire fișierul csv de pe Kaggle
try:
    df_real = pd.read_csv("weatherHistory.csv")
except FileNotFoundError:
    print("Eroare, nu exista !")
    exit()

# Convertire coloana text în obiecte datetime 
print("Convertim datele")
df_real['Formatted Date'] = pd.to_datetime(df_real['Formatted Date'], utc=True)
df_real['Ora'] = df_real['Formatted Date'].dt.hour

# Selectare si redenumire coloane pentru ML
df_clean = pd.DataFrame()
df_clean['Ora'] = df_real['Ora']
df_clean['Temperatura_Curenta'] = df_real['Temperature (C)']
df_clean['Umiditate'] = df_real['Humidity'] * 100  # Convertim 0.89 -> 89%
df_clean['Presiune'] = df_real['Pressure (millibars)']

# Target -> Temperatura peste 1 oră (shiftare cu -1)
df_clean['Temperatura_Viitoare'] = df_clean['Temperatura_Curenta'].shift(-1)

# eliminare ultima linie 
df_clean = df_clean.dropna()

# actualizare fisier
df_clean.to_csv("dataset_weather.csv", index=False)

print(f"Am transformat {len(df_clean)} rânduri de date reale.")
print("Acum fișierul 'dataset_weather.csv' conține date reale din Szeged.")
print("Rulează 'python ml_model.py' ca să re antrenezi modelul!")