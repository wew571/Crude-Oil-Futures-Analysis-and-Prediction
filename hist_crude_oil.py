
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

csv_path = Path("/Users/socolabingsu/Documents/Toan_roi_rac/Crude_Oil/data/Crude_Oil_Futures.csv")
df = pd.read_csv(csv_path)

print(df.head())

df.columns = [c.lower().strip() for c in df.columns]

# ---- Ép kiểu số cho các cột cần thiết ----
for col in ["open", "high", "low", "close", "volume"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---- Vẽ histogram ----
for col in ["open", "high", "low", "close", "volume"]:
    if col in df.columns:
        plt.figure()
        plt.hist(df[col].dropna(), bins=50)
        plt.title(f"Phân phối {col}")
        plt.xlabel(col)
        plt.ylabel("Tần suất")
        plt.grid(True, alpha=0.3)
        plt.show()
