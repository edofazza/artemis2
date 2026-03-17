import os

os.environ['TRANSFORMERS_CACHE'] = './cache'
os.environ['TORCH_HOME'] = './torch_cache'

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# Function to process frames in a folder
def extract_features_from_folder(folder_path, output_path, clip_transform):
    frame_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg")]
    frame_paths.sort()  # Ensure frames are processed in order
    os.makedirs(output_path, exist_ok=True)

    # Batch processing
    batch_size = 16  # Adjust batch size based on your hardware
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i+batch_size]
        images = [clip_transform(Image.open(p).convert("RGB")) for p in batch_paths]
        images = torch.stack(images).to(device)

        with torch.no_grad():
            # Extract features from CLIP's image encoder
            features = clip_model(images).pooler_output  # Use the pooled output
            features = features.cpu().numpy()

        # Save features
        for path, feature in zip(batch_paths, features):
            frame_name = os.path.splitext(os.path.basename(path))[0]
            feature_path = os.path.join(output_path, f"{frame_name}.npy")
            np.save(feature_path, feature)



if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CLIP model and processor
    model_name = "openai/clip-vit-base-patch32"  # You can use larger models if needed, e.g., "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name).vision_model
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Preprocessing using CLIP's processor
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_transform = Compose([
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # Directory paths
    input_dir = "AnimalKingdom/action_recognition/dataset/image/"  # Replace with your input folder path
    output_dir = "AnimalKingdom/action_recognition/features/clip"  # Replace with your output folder path
    os.makedirs(output_dir, exist_ok=True)

    # Process all folders
    for folder_name in tqdm(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder_name)
        if os.path.isdir(folder_path):
            output_path = os.path.join(output_dir, folder_name)
            extract_features_from_folder(folder_path, output_path, clip_transform)

    print("Feature extraction completed.")
