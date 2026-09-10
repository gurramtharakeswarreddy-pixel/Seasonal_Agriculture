"""
Seasonal Agriculture Performance Analysis
Author: [Your Name]
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "data/seasonal_agriculture_performance_dataset.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("SEASONAL AGRICULTURE PERFORMANCE ANALYSIS")
print("=" * 60)

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# -----------------------------
# 2. Data Cleaning
# -----------------------------
df = df.drop_duplicates()
df = df.dropna()

print("\nCleaned Dataset Shape:", df.shape)

# -----------------------------
# 3. Descriptive Statistics
# -----------------------------
print("\nDescriptive Statistics:")
print(df.describe())

# -----------------------------
# 4. Seasonal Performance
# -----------------------------
season_summary = df.groupby("Season").agg(
    Average_Yield=("Yield_Tonnes_Ha", "mean"),
    Average_Production=("Production_Tonnes", "mean"),
    Average_Profit=("Profit_INR", "mean"),
    Average_Rainfall=("Rainfall_mm", "mean"),
    Average_Water_Used=("Water_Used_m3", "mean"),
    Average_Water_Efficiency=("Water_Efficiency_t_per_1000m3", "mean")
).round(2)

print("\nSeasonal Performance Summary:")
print(season_summary)

season_summary.to_csv(
    os.path.join(OUTPUT_DIR, "seasonal_summary.csv")
)

# -----------------------------
# 5. Average Yield by Season
# -----------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Season", y="Yield_Tonnes_Ha", estimator=np.mean)
plt.title("Average Yield Across Seasons")
plt.xlabel("Season")
plt.ylabel("Average Yield (Tonnes/Ha)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "average_yield_by_season.png"))
plt.close()

# -----------------------------
# 6. Average Production by Season
# -----------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Season", y="Production_Tonnes", estimator=np.mean)
plt.title("Average Production Across Seasons")
plt.xlabel("Season")
plt.ylabel("Average Production (Tonnes)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "average_production_by_season.png"))
plt.close()

# -----------------------------
# 7. Average Profit by Season
# -----------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Season", y="Profit_INR", estimator=np.mean)
plt.title("Average Profit Across Seasons")
plt.xlabel("Season")
plt.ylabel("Average Profit (INR)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "average_profit_by_season.png"))
plt.close()

# -----------------------------
# 8. Rainfall by Season
# -----------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Season", y="Rainfall_mm", estimator=np.mean)
plt.title("Average Rainfall Across Seasons")
plt.xlabel("Season")
plt.ylabel("Average Rainfall (mm)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "average_rainfall_by_season.png"))
plt.close()

# -----------------------------
# 9. Rainfall vs Yield
# -----------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Rainfall_mm",
    y="Yield_Tonnes_Ha",
    hue="Season",
    alpha=0.6
)
plt.title("Relationship Between Rainfall and Yield")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Yield (Tonnes/Ha)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rainfall_vs_yield.png"))
plt.close()

# -----------------------------
# 10. Water Used vs Efficiency
# -----------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Water_Used_m3",
    y="Water_Efficiency_t_per_1000m3",
    hue="Season",
    alpha=0.6
)
plt.title("Water Usage vs Water Efficiency")
plt.xlabel("Water Used (m³)")
plt.ylabel("Water Efficiency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "water_usage_vs_efficiency.png"))
plt.close()

# -----------------------------
# 11. Crop-wise Yield
# -----------------------------
crop_yield = (
    df.groupby("Crop")["Yield_Tonnes_Ha"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Yield by Crop:")
print(crop_yield.round(2))

plt.figure(figsize=(10, 5))
crop_yield.plot(kind="bar")
plt.title("Average Yield by Crop")
plt.xlabel("Crop")
plt.ylabel("Yield (Tonnes/Ha)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "average_yield_by_crop.png"))
plt.close()

# -----------------------------
# 12. Irrigation Method Analysis
# -----------------------------
irrigation_profit = (
    df.groupby("Irrigation_Method")["Profit_INR"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Profit by Irrigation Method:")
print(irrigation_profit.round(2))

plt.figure(figsize=(8, 5))
irrigation_profit.plot(kind="bar")
plt.title("Average Profit by Irrigation Method")
plt.xlabel("Irrigation Method")
plt.ylabel("Average Profit (INR)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "profit_by_irrigation_method.png"))
plt.close()

# -----------------------------
# 13. Correlation Analysis
# -----------------------------
numeric_columns = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Soil_Moisture_pct",
    "Fertilizer_kg_ha",
    "Yield_Tonnes_Ha",
    "Production_Tonnes",
    "Profit_INR",
    "Water_Used_m3",
    "Water_Efficiency_t_per_1000m3"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Between Agricultural Variables")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# -----------------------------
# 14. Key Insights
# -----------------------------
best_yield_season = season_summary["Average_Yield"].idxmax()
best_profit_season = season_summary["Average_Profit"].idxmax()
highest_rainfall_season = season_summary["Average_Rainfall"].idxmax()

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"Highest average yield season: {best_yield_season}")
print(f"Highest average profit season: {best_profit_season}")
print(f"Highest average rainfall season: {highest_rainfall_season}")

print("\nAnalysis completed successfully.")
print(f"All charts and results are saved in the '{OUTPUT_DIR}' folder.")
