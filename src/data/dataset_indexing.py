"""Add index column to blendshapes test dataset for error analysis"""

import csv

INPUT_PATH = "data/blendshapes_test.csv"
OUTPUT_PATH = "data/blendshapes_test_indexed.csv"

rows = []

# Read dataset
with open(INPUT_PATH, encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

    for i, row in enumerate(reader):
        rows.append(row + [i])  # append index

# New header
new_header = header + ["index"]

# Write new file
with open(OUTPUT_PATH, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(new_header)
    writer.writerows(rows)

print(f"Saved indexed test dataset to {OUTPUT_PATH}")
print(f"Total rows: {len(rows)}")