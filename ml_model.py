import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Folosit pentru a salva modelul pe disc .pkl

# incarcarea datelor

print("Încărcăm datele...")
try:
    df = pd.read_csv("dataset_weather.csv")
except FileNotFoundError:
    print("Eroare: Fișierul 'dataset_weather.csv' nu există. Rulează generatorul de date întâi!")
    exit()

# Selectarea datelor 

# X (Input Ora, Temp curentă, Umiditate, Presiune)
X = df[['Ora', 'Temperatura_Curenta', 'Umiditate', 'Presiune']]

# y (Output -> Temperatura peste o ora)
y = df['Temperatura_Viitoare']

# 3. impărtirea datelor 20/80
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Antrenarea Modelului cu random forest (Arbori de Decizie) - e robust și nu cere scalarea datelor
print("Antrenăm modelul Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluarea Modelului
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"Model Antrenat cu succes!")
print(f"Eroare Medie Pătratică (MSE): {mse:.4f}")
print(f"Acuratețe (R2 Score): {r2:.4f} (1.0 înseamnă perfect)")
print("-" * 30)

# 6. Salvarea Modelului in format .pkl
joblib.dump(model, "model_clima.pkl")
print("Modelul a fost salvat în fișierul 'model_clima.pkl'")