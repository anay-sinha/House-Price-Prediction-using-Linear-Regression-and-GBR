"""
House Price Prediction
Using Linear Regression and Gradient Boosting Regressor
Dataset: house_prices.csv (Thane, India)
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("house_prices.csv")
print(f"Dataset shape: {df.shape}")
print(df.head(3))

# ─────────────────────────────────────────────
# 2. CLEAN & ENGINEER FEATURES
# ─────────────────────────────────────────────

# --- Target: Price (in rupees) per sq ft ---
# Drop rows where target is missing or zero
df = df[pd.to_numeric(df["Price (in rupees)"], errors="coerce").notna()]
df["price_per_sqft"] = pd.to_numeric(df["Price (in rupees)"], errors="coerce")
df = df[df["price_per_sqft"] > 0].copy()

# --- Convert Amount to numeric (in Lakhs) ---
def parse_amount(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    if "cr" in val:
        num = re.findall(r"[\d.]+", val)
        return float(num[0]) * 100 if num else np.nan
    elif "lac" in val or "lak" in val:
        num = re.findall(r"[\d.]+", val)
        return float(num[0]) if num else np.nan
    return np.nan

df["amount_lac"] = df["Amount(in rupees)"].apply(parse_amount)

# --- Carpet Area (in sqft) ---
def parse_sqft(val):
    if pd.isna(val):
        return np.nan
    num = re.findall(r"[\d.]+", str(val))
    return float(num[0]) if num else np.nan

df["carpet_sqft"] = df["Carpet Area"].apply(parse_sqft)
df["super_sqft"] = df["Super Area"].apply(parse_sqft)
# Use carpet area; fallback to super area
df["area_sqft"] = df["carpet_sqft"].combine_first(df["super_sqft"])

# --- Floor number (current floor) ---
def parse_floor(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower()
    if "ground" in val:
        return 0
    if "basement" in val:
        return -1
    num = re.findall(r"\d+", val)
    return float(num[0]) if num else np.nan

df["floor_num"] = df["Floor"].apply(parse_floor)

# --- BHK count extracted from title ---
def parse_bhk(title):
    match = re.search(r"(\d)\s*bhk", str(title).lower())
    return int(match.group(1)) if match else np.nan

df["bhk"] = df["Title"].apply(parse_bhk)

# --- Bathroom count ---
df["bathrooms"] = pd.to_numeric(df["Bathroom"], errors="coerce")

# --- Furnishing (ordinal: 0=Unfurnished, 1=Semi, 2=Furnished) ---
furnish_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
df["furnishing_enc"] = df["Furnishing"].map(furnish_map)

# --- Transaction type (Resale=0, New=1) ---
df["is_new"] = (df["Transaction"].str.lower().str.contains("new", na=False)).astype(int)

# --- Status (Ready to Move=1, Under Construction=0) ---
df["ready_to_move"] = (df["Status"].str.lower().str.contains("ready", na=False)).astype(int)

# ─────────────────────────────────────────────
# 3. SELECT FEATURES & DROP NULLS
# ─────────────────────────────────────────────
features = [
    "bhk",
    "area_sqft",
    "floor_num",
    "bathrooms",
    "furnishing_enc",
    "is_new",
    "ready_to_move",
    "amount_lac",
]

target = "price_per_sqft"

model_df = df[features + [target]].dropna()
print(f"\nUsable rows after cleaning: {len(model_df)}")

X = model_df[features]
y = model_df[target]

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)
gbr.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def evaluate(name, model, X_t, y_t):
    preds = model.predict(X_t)
    mae  = mean_absolute_error(y_t, preds)
    rmse = np.sqrt(mean_squared_error(y_t, preds))
    r2   = r2_score(y_t, preds)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")
    return preds

lr_preds  = evaluate("Linear Regression",          lr,  X_test, y_test)
gbr_preds = evaluate("Gradient Boosting Regressor", gbr, X_test, y_test)

# ─────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("House Price Prediction — Actual vs Predicted (₹/sqft)", fontsize=14)

for ax, preds, name, color in zip(
    axes,
    [lr_preds, gbr_preds],
    ["Linear Regression", "Gradient Boosting Regressor"],
    ["steelblue", "darkorange"],
):
    ax.scatter(y_test, preds, alpha=0.5, color=color, edgecolors="k", linewidths=0.3, s=30)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Price (₹/sqft)")
    ax.set_ylabel("Predicted Price (₹/sqft)")
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.savefig("predictions_scatter.png", dpi=150)
plt.close()

# --- Feature importance (GBR) ---
feat_imp = pd.Series(gbr.feature_importances_, index=features).sort_values()
fig2, ax2 = plt.subplots(figsize=(8, 4))
feat_imp.plot(kind="barh", color="darkorange", ax=ax2)
ax2.set_title("GBR Feature Importances")
ax2.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nPlots saved: predictions_scatter.png, feature_importance.png")
print("\nDone.")
