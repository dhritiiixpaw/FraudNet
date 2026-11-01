
import csv
from pathlib import Path

source = Path("data/raw/PaySim.csv")
destination = Path("data/demo_transactions.csv")

columns = ["type", "amount", "isFraud"]

with source.open("r", newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)

    with destination.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=columns)
        writer.writeheader()

        count = 0

        for row in reader:
            writer.writerow({
                "type": row["type"],
                "amount": row["amount"],
                "isFraud": row["isFraud"],
            })

            count += 1

            if count >= 100_000:
                break

print(f"Demo dataset created: {count:,} rows")
print(f"Saved to: {destination}")
