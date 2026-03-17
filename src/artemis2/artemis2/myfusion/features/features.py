import os
import torch

os.environ['TORCH_HOME'] = './torch_cache'

import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# Function to process frames in a folder
def extract_features_from_folder(folder_path, output_path):
    frame_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg")]
    frame_paths.sort()  # Ensure frames are processed in order
    os.makedirs(output_path, exist_ok=True)

    # Batch processing
    batch_size = 256  # Adjust batch size based on your hardware
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i+batch_size]
        images = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        images = torch.stack(images).to(device)

        with torch.no_grad():
            features = resnet50(images).squeeze()  # Remove unnecessary dimensions
            features = features.cpu().numpy()

        # Save features
        for j, (path, feature) in enumerate(zip(batch_paths, features)):
            frame_name = os.path.splitext(os.path.basename(path))[0]
            feature_path = os.path.join(output_path, f"{frame_name}.npy")
            np.save(feature_path, feature)


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained ResNet-50 model and remove the classification head
    resnet50 = models.resnet152(pretrained=True)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])  # Remove the FC layer
    resnet50 = resnet50.to(device)
    resnet50.eval()

    # Transform for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Directory paths
    input_dir = "AnimalKingdom/action_recognition/dataset/image/"  # Replace with your input folder path
    output_dir = "AnimalKingdom/action_recognition/features/resnet152"  # Replace with your output folder path
    os.makedirs(output_dir, exist_ok=True)
    # Process all folders
    for folder_name in tqdm(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder_name)
        if os.path.isdir(folder_path):
            output_path = os.path.join(output_dir, folder_name)
            extract_features_from_folder(folder_path, output_path)

    print("Feature extraction completed.")