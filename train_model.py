import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import joblib


# ============================================================
# 1. CARICAMENTO DATASET
# ============================================================

df = pd.read_excel("dataset_case.xlsx")

print("Dataset caricato correttamente!")
print(f"Numero di righe: {len(df)}")
print(f"Numero di colonne: {len(df.columns)}")


# ============================================================
# 2. PREPARAZIONE DELLA VARIABILE GIARDINO
# ============================================================

df["Giardino_bin"] = df["Giardino"].map({
    "No": 0,
    "Sì": 1
})


# Controlliamo che la trasformazione sia avvenuta correttamente
print("\nValori Giardino:")
print(df[["Giardino", "Giardino_bin"]].head())


# ============================================================
# 3. DEFINIZIONE X E y
# ============================================================

X = df[
    [
        "Metri_Quadri",
        "Numero_Bagni",
        "Giardino_bin"
    ]
]

y = df["Prezzo_Casa"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nSuddivisione dataset:")
print(f"Training set: {len(X_train)} osservazioni")
print(f"Test set: {len(X_test)} osservazioni")


# ============================================================
# 5. ADDESTRAMENTO MODELLO
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)


print("\nModello addestrato correttamente!")


# ============================================================
# 6. PREDIZIONI SUL TEST SET
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. METRICHE
# ============================================================

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

n = len(y_test)
p = X_test.shape[1]

r2_adj = 1 - (
    (1 - r2) * (n - 1)
    / (n - p - 1)
)

rmse_relative = (
    rmse / np.mean(y_test)
) * 100


print("\n==========================================")
print("RISULTATI DEL MODELLO")
print("==========================================")

print(f"R²              = {r2:.4f}")
print(f"R² adjusted     = {r2_adj:.4f}")
print(f"RMSE            = {rmse:,.2f} €")
print(f"RMSE relativo   = {rmse_relative:.2f}%")


# ============================================================
# 8. COEFFICIENTI DEL MODELLO
# ============================================================

print("\n==========================================")
print("COEFFICIENTI")
print("==========================================")

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:,.2f}")

print(f"Intercept: {model.intercept_:,.2f} €")


# ============================================================
# 9. SALVATAGGIO DEL MODELLO
# ============================================================

joblib.dump(model, "model.pkl")

print("\n==========================================")
print("MODELLO SALVATO")
print("==========================================")

print("File creato: model.pkl")