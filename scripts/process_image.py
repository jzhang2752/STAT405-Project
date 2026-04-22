import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms


def load_model():

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    feature_model = torch.nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_model.eval()

    return feature_model


def preprocess():

    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])


def main():

    image_path = Path(sys.argv[1])

    output_dir = Path(sys.argv[2])

    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()

    transform = preprocess()

    img = Image.open(image_path).convert("RGB")

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():

        embedding = model(tensor)

    embedding = embedding.squeeze().numpy()

    with torch.no_grad():

        embedding = model(tensor)

    embedding = embedding.squeeze().numpy()

    city_name = image_path.parent.parent.name
    tile_name = image_path.parent.name
    save_name = f"{city_name}_{tile_name}_{image_path.stem}.npy"
    save_path = output_dir / save_name

    np.save(save_path, embedding)

    print("saved:", save_path)

    np.save(save_path, embedding)

    print("saved:", save_path)


if __name__ == "__main__":

    main()
