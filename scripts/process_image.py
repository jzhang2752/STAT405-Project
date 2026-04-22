import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms


def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_model = torch.nn.Sequential(*list(model.children())[:-1])
    feature_model.eval()
    return feature_model


def preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def main():
    if len(sys.argv) != 2:
        print("Usage: python process_image.py <image_file>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        sys.exit(1)

    model = load_model()
    transform = preprocess()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor)

    embedding_array = embedding.squeeze().cpu().numpy()

    output_name = image_path.stem + ".npy"
    output_path = Path(output_name)

    np.save(output_path, embedding_array)

    print(f"Saved embedding to {output_path}")


if __name__ == "__main__":
    main()
