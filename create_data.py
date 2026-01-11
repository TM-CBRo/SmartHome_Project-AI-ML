import pandas as pd
import numpy as np

# Configurare simulare
np.random.seed(42) 
dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
n = len(dates)

# Generarea datelor
hour_factor = 2 * np.pi * (dates.hour - 10) / 24
temp_base = 5 + 5 * np.sin(hour_factor) + np.random.normal(0, 1.5, n)
cold_front = np.linspace(0, -5, n)
temperature = temp_base + cold_front

humidity = 60 - (temperature * 2) + np.random.normal(0, 5, n)
humidity = np.clip(humidity, 30, 95)

# Creare DataFrame
df = pd.DataFrame({
    'Ora': dates.hour,
    'Temperatura_Curenta': temperature.round(1),
    'Umiditate': humidity.round(1),
    'Presiune': np.random.normal(1013, 5, n).round(1)
})

# Targetare (Ce prezicem: Temp peste 1 ora)
df['Temperatura_Viitoare'] = df['Temperatura_Curenta'].shift(-1)
df = df.dropna()

# Salvare
df.to_csv("dataset_weather.csv", index=False)
print("Fisier 'dataset_weather.csv' creat cu succes.")