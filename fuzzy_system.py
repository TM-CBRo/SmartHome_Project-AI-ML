import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def get_fuzzy_system():
    # Definirea Variabilelor (Antecedents & Consequents)

    
    # Diferenta temperatura (Temp Dorită - Temp Prezisa) -> -10 grade  la +10 grade 
    temp_diff = ctrl.Antecedent(np.arange(-10, 11, 1), 'temp_diff')

    #  Umiditatea (citită din senzor/date) -> 0% la 100%
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')

    # Puterea Sistemului de Climatizare -> -100% (Răcire Max) la +100% (Încălzire Max)
    power = ctrl.Consequent(np.arange(-100, 101, 1), 'power')

    # Definirea Funcțiilor si convertirea nr in cuvinte
    
    # Diferenta: Negativă (cald), Zero (ok), Pozitivă (frig)
    temp_diff['negative'] = fuzz.trimf(temp_diff.universe, [-10, -10, 0])
    temp_diff['zero']     = fuzz.trimf(temp_diff.universe, [-2, 0, 2])
    temp_diff['positive'] = fuzz.trimf(temp_diff.universe, [0, 10, 10])

    # Umiditate: Uscat, Optim, Umed
    humidity['dry']       = fuzz.trimf(humidity.universe, [0, 0, 40])
    humidity['optimal']   = fuzz.trimf(humidity.universe, [30, 50, 70])
    humidity['wet']       = fuzz.trimf(humidity.universe, [60, 100, 100])

    # Putere: Răcire, Oprit, Încălzire
    power['cooling']      = fuzz.trimf(power.universe, [-100, -100, -10])
    power['off']          = fuzz.trimf(power.universe, [-20, 0, 20])
    power['heating']      = fuzz.trimf(power.universe, [10, 100, 100])

    # Reguli IF-THEN -> IA
    rule1 = ctrl.Rule(temp_diff['negative'], power['cooling'])

    # Dacă e mai frig decât vremea (diff poz), pornește caldura
    rule2 = ctrl.Rule(temp_diff['positive'], power['heating'])

    # Dacă temp e ok (zero), oprește totul
    rule3 = ctrl.Rule(temp_diff['zero'], power['off'])

    # Dacă e frig (poz) si e foarte umed, porneste caldura mai tare
    # (pentru că umiditatea face frigul să se resimta mai rau)
    rule4 = ctrl.Rule(temp_diff['positive'] & humidity['wet'], power['heating'])

    # Crearea Sistemului de Control

    climate_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    simulation = ctrl.ControlSystemSimulation(climate_ctrl)
    
    return simulation

if __name__ == "__main__":
    print("Testing sistemul Fuzzy...")
    
    # creare simulatorul
    sim = get_fuzzy_system()

    target_temp = 24
    predicted_temp = 18
    diff = target_temp - predicted_temp
    
    current_humidity = 75 
    
    print(f"Scenariu: Diferența={diff}°C, Umiditate={current_humidity}%")
    
    # Introducere datele in sistem
    sim.input['temp_diff'] = diff
    sim.input['humidity'] = current_humidity
    
    sim.compute()
    
    result = sim.output['power']
    print(f"Decizia AI: Putere Sistem = {result:.2f}%")
    
    if result > 0:
        print("Centrala PORNEȘTE (Încălzire)")
    elif result < 0:
        print("Aerul Condiționat PORNEȘTE (Răcire)")
    else:
        print("Sistem OPRIT")