
import tensorflow_datasets as tfds
import pandas as pd
from pathlib import Path

# Create the data folder
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Download and load Hillstrom
dataset = tfds.load(
    "hillstrom",
    split="train",
    as_supervised=False
)

# Convert the records into a DataFrame
df = tfds.as_dataframe(dataset)

# Save as a CSV file
output_file = data_folder / "hillstrom.csv"
df.to_csv(output_file, index=False)

print("Download completed!")
print("Dataset shape:", df.shape)
print("Saved to:", output_file)
print(df.head())
