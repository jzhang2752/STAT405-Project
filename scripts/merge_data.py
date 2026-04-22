import pandas as pd

EMBED_PATH = "../outputs/image_embeddings_table.csv"
HOUSE_PATH = "../HouseTS.csv"
OUT_PATH = "../outputs/final_dataset_small.csv"

# Read embeddings
embeddings = pd.read_csv(EMBED_PATH)

# Keep only key columns + first 20 image features
image_feature_columns = [column for column in embeddings.columns if column.startswith("img_feat_")]
image_feature_columns = image_feature_columns[:20]

embeddings = embeddings[["location_id", "year"] + image_feature_columns].copy()

# Normalize keys
embeddings["zipcode"] = embeddings["location_id"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
embeddings["year"] = embeddings["year"].astype(int)

# Read only the HouseTS columns we need
house_columns = [
    "zipcode",
    "year",
    "city",
    "city_full",
    "price",
    "Median Home Value",
    "median_sale_price",
    "median_list_price",
    "homes_sold",
    "inventory"
]

house = pd.read_csv(
    HOUSE_PATH,
    usecols=house_columns,
    dtype={"zipcode": str}
)

house["zipcode"] = house["zipcode"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
house["year"] = house["year"].astype(int)

# Merge only rows that have embeddings
merged = house.merge(
    embeddings.drop(columns=["location_id"]),
    on=["zipcode", "year"],
    how="inner"
)

merged.to_csv(OUT_PATH, index=False)

print("House shape:", house.shape)
print("Embeddings shape:", embeddings.shape)
print("Merged shape:", merged.shape)
print("Saved to:", OUT_PATH)
