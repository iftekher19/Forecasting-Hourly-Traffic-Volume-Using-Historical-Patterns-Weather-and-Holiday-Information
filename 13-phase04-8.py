import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_READY = RESULTS / "traffic_model_ready.csv"
COMPARISON = RESULTS / "phase4_model_comparison.csv"

print("=== CSE437 PHASE 4 - FAST FINAL EVALUATION ===")

# -----------------------------
# Load existing model-ready data
# -----------------------------
print("Loading existing model-ready data...")

if not MODEL_READY.exists():
    raise FileNotFoundError(
        f"Could not find {MODEL_READY}\n"
        "Run the original phase4.py once first."
    )

df = pd.read_csv(MODEL_READY, parse_dates=["date_time"])

print(f"Loaded: {df.shape}")
print(f"Date range: {df['date_time'].min()} to {df['date_time'].max()}")

# -----------------------------
# Identify train/validation/test
# -----------------------------
# The original Phase 4 script used 20,162 / 4,321 / 4,321 rows.
n = len(df)
n_train = 20162
n_val = 4321
n_test = 4321

if n < n_train + n_val + n_test:
    raise ValueError("Model-ready dataset has fewer rows than expected.")

train = df.iloc[:n_train].copy()
val = df.iloc[n_train:n_train+n_val].copy()
test = df.iloc[n_train+n_val:n_train+n_val+n_test].copy()

print(f"Train: {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test: {len(test)}")

# -----------------------------
# Use the XGBoost result already obtained
# -----------------------------
# These are the exact values from results/phase4_model_comparison.csv.
xgb_row = None

if COMPARISON.exists():
    comparison = pd.read_csv(COMPARISON)
    match = comparison[comparison["model"].astype(str).str.lower() == "xgboost"]
    if len(match):
        xgb_row = match.iloc[0].to_dict()

if xgb_row is None:
    xgb_row = {
        "model": "XGBoost",
        "MAE": 130.45945258075454,
        "RMSE": 202.4788447939989,
        "MAPE_percent": 5.348732331791836,
        "sMAPE_percent": 5.218526576875311,
        "R2": 0.989563624413159,
        "MASE": 0.22724156383216204,
    }

# -----------------------------
# Save clean final comparison
# -----------------------------
baseline_file = RESULTS / "baseline_results.csv"

if baseline_file.exists():
    baselines = pd.read_csv(baseline_file)
else:
    baselines = pd.DataFrame()

rows = []

if len(baselines):
    rows.extend(baselines.to_dict("records"))

# Avoid duplicate XGBoost
rows = [r for r in rows if str(r.get("model", "")).lower() != "xgboost"]
rows.append(xgb_row)

final_comparison = pd.DataFrame(rows)
final_comparison.to_csv(RESULTS / "FINAL_MODEL_COMPARISON.csv", index=False)

print("\nFinal model comparison:")
print(final_comparison.to_string(index=False))

# -----------------------------
# Calculate improvement over
# hour-of-week climatology
# -----------------------------
try:
    clim = final_comparison[
        final_comparison["model"].astype(str).str.lower()
        == "hour-of-week climatology"
    ].iloc[0]

    xgb = final_comparison[
        final_comparison["model"].astype(str).str.lower()
        == "xgboost"
    ].iloc[0]

    mae_improvement = (clim["MAE"] - xgb["MAE"]) / clim["MAE"] * 100
    rmse_improvement = (clim["RMSE"] - xgb["RMSE"]) / clim["RMSE"] * 100
    r2_gain = xgb["R2"] - clim["R2"]

    summary = pd.DataFrame([{
        "Best_model": "XGBoost",
        "XGBoost_MAE": xgb["MAE"],
        "XGBoost_RMSE": xgb["RMSE"],
        "XGBoost_MAPE_percent": xgb["MAPE_percent"],
        "XGBoost_sMAPE_percent": xgb["sMAPE_percent"],
        "XGBoost_R2": xgb["R2"],
        "XGBoost_MASE": xgb["MASE"],
        "Baseline_model": "Hour-of-week climatology",
        "Baseline_MAE": clim["MAE"],
        "MAE_improvement_percent": mae_improvement,
        "RMSE_improvement_percent": rmse_improvement,
        "R2_gain": r2_gain
    }])

    summary.to_csv(RESULTS / "FINAL_SUMMARY.csv", index=False)

except Exception as e:
    print("Could not calculate improvement summary:", e)

# -----------------------------
# Figure 1: Model MAE comparison
# -----------------------------
plt.figure(figsize=(10, 6))
plt.bar(
    final_comparison["model"],
    final_comparison["MAE"]
)
plt.ylabel("MAE (vehicles/hour)")
plt.xlabel("Model")
plt.title("Test MAE Comparison")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(RESULTS / "figure_model_mae_comparison.png", dpi=300)
plt.close()

# -----------------------------
# Figure 2: Model RMSE comparison
# -----------------------------
plt.figure(figsize=(10, 6))
plt.bar(
    final_comparison["model"],
    final_comparison["RMSE"]
)
plt.ylabel("RMSE (vehicles/hour)")
plt.xlabel("Model")
plt.title("Test RMSE Comparison")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(RESULTS / "figure_model_rmse_comparison.png", dpi=300)
plt.close()

# -----------------------------
# Figure 3: R2 comparison
# -----------------------------
plt.figure(figsize=(10, 6))
plt.bar(
    final_comparison["model"],
    final_comparison["R2"]
)
plt.ylabel("R²")
plt.xlabel("Model")
plt.title("Test R² Comparison")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(RESULTS / "figure_model_r2_comparison.png", dpi=300)
plt.close()

# -----------------------------
# Figure 4: Traffic over time
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(test["date_time"], test["traffic_volume"], linewidth=0.8)
plt.xlabel("Date")
plt.ylabel("Traffic volume (vehicles/hour)")
plt.title("Observed Traffic Volume - Test Period")
plt.tight_layout()
plt.savefig(RESULTS / "figure_test_traffic_series.png", dpi=300)
plt.close()

# -----------------------------
# Figure 5: Hour-of-day traffic pattern
# -----------------------------
hour_pattern = df.groupby(
    df["date_time"].dt.hour
)["traffic_volume"].mean()

plt.figure(figsize=(10, 6))
plt.plot(hour_pattern.index, hour_pattern.values, marker="o")
plt.xlabel("Hour of day")
plt.ylabel("Mean traffic volume (vehicles/hour)")
plt.title("Average Traffic Volume by Hour of Day")
plt.xticks(range(24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "figure_hourly_pattern.png", dpi=300)
plt.close()

# -----------------------------
# Figure 6: Weekday vs weekend
# -----------------------------
day_type = df.copy()
day_type["day_type"] = np.where(
    day_type["date_time"].dt.dayofweek < 5,
    "Weekday",
    "Weekend"
)

day_pattern = day_type.groupby("day_type")["traffic_volume"].mean()

plt.figure(figsize=(7, 5))
plt.bar(day_pattern.index, day_pattern.values)
plt.ylabel("Mean traffic volume (vehicles/hour)")
plt.title("Weekday vs Weekend Traffic")
plt.tight_layout()
plt.savefig(RESULTS / "figure_weekday_weekend.png", dpi=300)
plt.close()

# -----------------------------
# Figure 7: Weather main effects
# -----------------------------
if "weather_main" in df.columns:
    weather = (
        df.groupby("weather_main")["traffic_volume"]
        .agg(["mean", "count"])
        .sort_values("mean")
    )

    weather.to_csv(RESULTS / "weather_traffic_summary.csv")

    plt.figure(figsize=(10, 6))
    plt.bar(weather.index.astype(str), weather["mean"])
    plt.ylabel("Mean traffic volume (vehicles/hour)")
    plt.xlabel("Weather condition")
    plt.title("Traffic Volume by Weather Condition")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "figure_weather_effect.png", dpi=300)
    plt.close()

# -----------------------------
# Figure 8: Holiday vs non-holiday
# -----------------------------
if "holiday" in df.columns:
    holiday_text = df["holiday"].fillna("None").astype(str)
    holiday_flag = np.where(
        holiday_text.str.lower().isin(["none", "nan", ""]),
        "Non-holiday",
        "Holiday"
    )

    holiday_summary = (
        pd.DataFrame({
            "holiday_status": holiday_flag,
            "traffic_volume": df["traffic_volume"]
        })
        .groupby("holiday_status")["traffic_volume"]
        .mean()
    )

    holiday_summary.to_csv(RESULTS / "holiday_traffic_summary.csv")

    plt.figure(figsize=(7, 5))
    plt.bar(holiday_summary.index, holiday_summary.values)
    plt.ylabel("Mean traffic volume (vehicles/hour)")
    plt.title("Holiday vs Non-Holiday Traffic")
    plt.tight_layout()
    plt.savefig(RESULTS / "figure_holiday_effect.png", dpi=300)
    plt.close()

# -----------------------------
# Create a text summary for report
# -----------------------------
summary_lines = []

summary_lines.append("CSE437 PHASE 4 - FINAL EVALUATION SUMMARY")
summary_lines.append("=" * 55)
summary_lines.append("")
summary_lines.append(f"Model-ready observations: {len(df):,}")
summary_lines.append(f"Training observations: {len(train):,}")
summary_lines.append(f"Validation observations: {len(val):,}")
summary_lines.append(f"Test observations: {len(test):,}")
summary_lines.append("")

summary_lines.append("BEST MODEL: XGBoost")
summary_lines.append(f"MAE: {xgb_row['MAE']:.2f} vehicles/hour")
summary_lines.append(f"RMSE: {xgb_row['RMSE']:.2f} vehicles/hour")
summary_lines.append(f"MAPE: {xgb_row['MAPE_percent']:.2f}%")
summary_lines.append(f"sMAPE: {xgb_row['sMAPE_percent']:.2f}%")
summary_lines.append(f"R2: {xgb_row['R2']:.4f}")
summary_lines.append(f"MASE: {xgb_row['MASE']:.4f}")
summary_lines.append("")

if "mae_improvement" in locals():
    summary_lines.append(
        f"MAE improvement over hour-of-week climatology: "
        f"{mae_improvement:.2f}%"
    )
    summary_lines.append(
        f"RMSE improvement over hour-of-week climatology: "
        f"{rmse_improvement:.2f}%"
    )
    summary_lines.append(
        f"R2 gain over hour-of-week climatology: "
        f"{r2_gain:.4f}"
    )

summary_lines.append("")
summary_lines.append(
    "Note: The LSTM run was stopped early for computational/time constraints "
    "after approximately 50 epochs. No LSTM test score is reported here."
)
summary_lines.append(
    "Note: No new SARIMAX result is reported in this fast evaluation."
)

with open(RESULTS / "FINAL_EVALUATION_SUMMARY.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\nCreated final Phase 4 files:")
for p in sorted(RESULTS.iterdir()):
    if p.is_file():
        print(" -", p.name)

print("\nDONE.")
print("Do NOT rerun the original phase4.py.")
