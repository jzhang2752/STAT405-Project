from pathlib import Path
import numpy as np
import pandas as pd

EMBEDDING_DIR = Path("../embeddings")
OUTPUT_FILE = Path("../outputs/image_embeddings_table.csv")

def parse_file_name(file_name: str):
    stem = Path(file_name).stem
    parts = stem.split("_")

    if len(parts) < 2:
        return None, None

    year = parts[-1]
    location_id = "_".join(parts[:-1])

    return location_id, year


def main():

    rows = []

    for npy_file in sorted(EMBEDDING_DIR.glob("*.npy")):

        location_id, year = parse_file_name(npy_file.name)

        if location_id is None:
            continue

        embedding = np.load(npy_file)

        row = {
            "embedding_file": npy_file.name,
            "location_id": location_id,
            "year": year
        }

        for i, value in enumerate(embedding):

            row[f"img_feat_{i}"] = float(value)

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_FILE, index=False)

    print("rows created:", len(df))
    print("saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
