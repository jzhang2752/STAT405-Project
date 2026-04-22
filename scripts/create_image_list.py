from pathlib import Path

IMAGE_ROOT = Path("../images/house_images/Housets_Final")
OUTPUT_FILE = Path("../outputs/images_list.txt")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

def main():

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for path in IMAGE_ROOT.rglob("*"):
        if path.suffix.lower() in VALID_EXTENSIONS:
            image_paths.append(path.resolve())

    with open(OUTPUT_FILE, "w") as f:
        for p in image_paths:
            f.write(str(p) + "\n")

    print("total images:", len(image_paths))


if __name__ == "__main__":
    main()
