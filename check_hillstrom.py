
import pandas as pd

# Load the dataset
df = pd.read_csv("data/hillstrom.csv")

# Display basic information
print("Dataset loaded successfully!")
print("Number of rows and columns:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nCustomers in each experiment group:")
print(df["segment"].value_counts())

print("\nTotal website visits:")
print(df["visit"].sum())

print("\nTotal conversions:")
print(df["conversion"].sum())

print("\nTotal spending:")
print(df["spend"].sum())

# Compare results across experiment groups
results = df.groupby("segment").agg(
    customers=("conversion", "count"),
    conversions=("conversion", "sum"),
    visits=("visit", "sum"),
    total_spend=("spend", "sum")
)

# Calculate rates and average spending
results["conversion_rate_pct"] = (
    results["conversions"] / results["customers"] * 100
)

results["visit_rate_pct"] = (
    results["visits"] / results["customers"] * 100
)

results["avg_spend_per_customer"] = (
    results["total_spend"] / results["customers"]
)

print("\nExperiment Results:")
print(results.round(2))
print("\nConversion rate by group:")
print(results[["customers", "conversions", "conversion_rate_pct"]].to_string())