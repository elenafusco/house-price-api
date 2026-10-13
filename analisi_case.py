# ============================================================
# HOUSE PRICE DEMO - REGRESSION ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import statsmodels.api as sm


# ============================================================
# 1. DATASET
# ============================================================
# Lettura del file Excel
df = pd.read_excel("dataset_case.xlsx")

print("DATASET")
print(df.head())

print("\nDimensioni dataset:", df.shape)

print("\nVariabili:")
print(df.columns)

print("\nInformazioni dataset:")
print(df.info())


# ============================================================
# 2. BOX PLOT + SINGOLI DATI: PREZZO / NUMERO BAGNI
# ============================================================

df["Prezzo_per_Bagno"] = df["Prezzo_Casa"] / df["Numero_Bagni"]

plt.figure(figsize=(6, 5))

# Boxplot
sns.boxplot(
    y=df["Prezzo_per_Bagno"],
    width=0.4
)

# Singole osservazioni
sns.stripplot(
    y=df["Prezzo_per_Bagno"],
    color="black",
    size=6,
    jitter=0.15
)

plt.title("Prezzo / Numero di bagni")
plt.ylabel("Prezzo per bagno (€)")
plt.grid(axis="y", alpha=0.3)

plt.show()


# ============================================================
# 3. BOX PLOT + SINGOLI DATI: PREZZO / METRI QUADRI
# ============================================================

df["Prezzo_per_m2"] = df["Prezzo_Casa"] / df["Metri_Quadri"]

plt.figure(figsize=(6, 5))

# Boxplot
sns.boxplot(
    y=df["Prezzo_per_m2"],
    width=0.4
)

# Singole osservazioni
sns.stripplot(
    y=df["Prezzo_per_m2"],
    color="black",
    size=6,
    jitter=0.15
)

plt.title("Prezzo / Metri quadri")
plt.ylabel("Prezzo per m² (€)")
plt.grid(axis="y", alpha=0.3)

plt.show()

# ============================================================
# 4. REGRESSIONE UNIVARIATA:
#    PREZZO ~ NUMERO BAGNI
# ============================================================

X = df[["Numero_Bagni"]]
y = df["Prezzo_Casa"]

model_bagni = LinearRegression()
model_bagni.fit(X, y)

y_pred_bagni = model_bagni.predict(X)

r2_bagni = r2_score(y, y_pred_bagni)

print("\n==========================================")
print("REGRESSIONE: PREZZO ~ NUMERO BAGNI")
print("==========================================")

print(f"Intercept: {model_bagni.intercept_:,.2f} €")
print(f"Coefficiente: {model_bagni.coef_[0]:,.2f} €")
print(f"R²: {r2_bagni:.4f}")


# Grafico
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Numero_Bagni"],
    y,
    label="Osservazioni"
)

# linea di regressione
x_line = np.linspace(
    df["Numero_Bagni"].min(),
    df["Numero_Bagni"].max(),
    100
).reshape(-1, 1)

y_line = model_bagni.predict(x_line)

plt.plot(
    x_line,
    y_line,
    linewidth=2,
    label=f"Regressione (R² = {r2_bagni:.3f})"
)

plt.xlabel("Numero di bagni")
plt.ylabel("Prezzo casa (€)")
plt.title("Regressione univariata: Prezzo vs Numero di bagni")
plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ============================================================
# 5. REGRESSIONE UNIVARIATA:
#    PREZZO ~ METRI QUADRI
# ============================================================

X = df[["Metri_Quadri"]]

model_m2 = LinearRegression()
model_m2.fit(X, y)

y_pred_m2 = model_m2.predict(X)

r2_m2 = r2_score(y, y_pred_m2)

print("\n==========================================")
print("REGRESSIONE: PREZZO ~ METRI QUADRI")
print("==========================================")

print(f"Intercept: {model_m2.intercept_:,.2f} €")
print(f"Coefficiente: {model_m2.coef_[0]:,.2f} € / m²")
print(f"R²: {r2_m2:.4f}")


# Grafico
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Metri_Quadri"],
    y,
    label="Osservazioni"
)

x_line = np.linspace(
    df["Metri_Quadri"].min(),
    df["Metri_Quadri"].max(),
    100
).reshape(-1, 1)

y_line = model_m2.predict(x_line)

plt.plot(
    x_line,
    y_line,
    linewidth=2,
    label=f"Regressione (R² = {r2_m2:.3f})"
)

plt.xlabel("Metri quadri")
plt.ylabel("Prezzo casa (€)")
plt.title("Regressione univariata: Prezzo vs Metri quadri")
plt.legend()
plt.grid(alpha=0.3)

plt.show()


# ============================================================
# 6. REGRESSIONE LINEARE MULTIVARIATA
# ============================================================

# Trasformiamo Giardino in variabile numerica:
# No = 0
# Sì = 1

df["Giardino_bin"] = df["Giardino"].map({
    "No": 0,
    "Sì": 1
})

X = df[
    [
        "Metri_Quadri",
        "Numero_Bagni",
        "Giardino_bin"
    ]
]

y = df["Prezzo_Casa"]


# ------------------------------------------------------------
# Train / Test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Modello
model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)


# ============================================================
# 7. METRICHE DEL MODELLO
# ============================================================

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

# R² adjusted
n = len(y_test)
p = X_test.shape[1]

r2_adj = 1 - (
    (1 - r2) * (n - 1)
    / (n - p - 1)
)
# RMSE relativo (%)
rmse_relative = (rmse / np.mean(y_test)) * 100

print("\n==========================================")
print("REGRESSIONE LINEARE MULTIVARIATA")
print("==========================================")

print("\nCoefficienti:")

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:,.2f}")

print(f"\nIntercept: {model.intercept_:,.2f} €")

print("\nMetriche sul TEST SET:")
print(f"R²       = {r2:.4f}")
print(f"R² adj   = {r2_adj:.4f}")
print(f"RMSE     = {rmse:,.2f} €")
print(f"RMSE relativo = {rmse_relative:.2f}%")

# ============================================================
# 8. OBSERVED VS PREDICTED
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    y_pred,
    s=70
)

# Linea ideale y = x
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Prezzo osservato (€)")
plt.ylabel("Prezzo predetto (€)")

plt.title(
    f"Observed vs Predicted\n"
    f"R² = {r2:.3f} | RMSE = {rmse:,.0f} € | RMSE relativo = {rmse_relative:.2f}%"
)

plt.grid(alpha=0.3)

plt.show()


# ============================================================
# 9. SUMMARY DEI RISULTATI
# ============================================================

print("\n==========================================")
print("SUMMARY")
print("==========================================")

print(f"R² univariata - Bagni:       {r2_bagni:.4f}")
print(f"R² univariata - Metri quadri:{r2_m2:.4f}")
print(f"R² multivariata:             {r2:.4f}")
print(f"R² adjusted:                 {r2_adj:.4f}")
print(f"RMSE:                        {rmse:,.2f} €")
print(f"RMSE relativo = {rmse_relative:.2f}%")

# ============================================================
# 10. NEW PREDICTION
# ============================================================

# Esempio:
# casa di 150 m²
# 2 bagni
# con giardino

new_house = pd.DataFrame({
    "Metri_Quadri": [150],
    "Numero_Bagni": [2],
    "Giardino_bin": [1]
})

new_prediction = model.predict(new_house)

print("\n==========================================")
print("NEW HOUSE PREDICTION")
print("==========================================")

print(
    f"Casa: 150 m², 2 bagni, giardino = Sì"
)

print(
    f"Prezzo predetto: {new_prediction[0]:,.2f} €"
)