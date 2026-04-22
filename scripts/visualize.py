import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../outputs/final_dataset_small.csv")

print("Columns:")
print(df.columns.tolist())
print("Shape:", df.shape)

if "Median Home Value" in df.columns:
    plt.figure()
    df["Median Home Value"].dropna().hist(bins=30)
    plt.title("Median Home Value Distribution")
    plt.xlabel("Median Home Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("../outputs/median_home_value_distribution.png")
    print("Saved ../outputs/median_home_value_distribution.png")

if "price" in df.columns:
    plt.figure()
    df["price"].dropna().hist(bins=30)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("../outputs/price_distribution.png")
    print("Saved ../outputs/price_distribution.png")
