
from pathlib import Path
import sys

import pandas as pd

# Make the src folder accessible to Python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from statistics_engine import compare_conversion_rates

# Load the actual dataset
df = pd.read_csv(PROJECT_ROOT / "data" / "hillstrom.csv")

# Select the Men's Email and No Email groups
mens = df[df["segment"] == "Mens E-Mail"]
control = df[df["segment"] == "No E-Mail"]

# Run our reusable statistical function
results = compare_conversion_rates(
    conversions_a=int(mens["conversion"].sum()),
    visitors_a=len(mens),
    conversions_b=int(control["conversion"].sum()),
    visitors_b=len(control),
)

# Display the results
print("Men's Email vs No Email")
print("-" * 35)

for metric, value in results.items():
    print(f"{metric}: {value}")
