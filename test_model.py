import joblib
import pandas as pd


# ============================================================
# 1. CARICAMENTO DEL MODELLO
# ============================================================

model = joblib.load("model.pkl")

print("Modello caricato correttamente!")


# ============================================================
# 2. NUOVA CASA
# ============================================================

nuova_casa = pd.DataFrame({
    "Metri_Quadri": [150],
    "Numero_Bagni": [2],
    "Giardino_bin": [1]
})


# ============================================================
# 3. PREVISIONE
# ============================================================

prezzo_predetto = model.predict(nuova_casa)


# ============================================================
# 4. RISULTATO
# ============================================================

print("\n==========================================")
print("PREVISIONE NUOVA CASA")
print("==========================================")

print(f"Metri quadri: 150")
print(f"Numero bagni: 2")
print(f"Giardino: Sì")

print(f"\nPrezzo stimato: {prezzo_predetto[0]:,.2f} €")