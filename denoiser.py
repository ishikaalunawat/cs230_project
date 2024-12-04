import os
import torch
from PIL import Image
from torchvision import transforms
from models_inspiration import MotionAwareDenoiser 

# Configuration
data_dir = "datasets_noisy/aquarium-data-cots/aquarium_pretrain"
model_path = "outputs_/denoiser_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
resize_to_model = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize for the model
    transforms.ToTensor()
])
resize_to_original = transforms.Compose([
    transforms.Resize((1024, 768))  # Resize back to original size
])
to_pil = transforms.ToPILImage()

# Load the model
model = MotionAwareDenoiser(in_channels=3, out_channels=3, num_features=64, num_blocks=8).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def process_and_save_images(data_split):
    """
    Process and save denoised images for a given data split (train, valid, test).

    Args:
        data_split (str): The data split (e.g., 'train', 'valid', 'test').
    """
    input_folder = os.path.join(data_dir, data_split, "noisy_images")
    output_folder = os.path.join(data_dir, data_split, "denoised_imgs")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all images in the input folder
    for img_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        # Open the image
        img = Image.open(input_path).convert("RGB")

        # Preprocess for the model
        img_tensor = resize_to_model(img).unsqueeze(0).to(device)

        # Pass through the model
        with torch.no_grad():
            denoised_tensor = model(img_tensor).squeeze(0).cpu()

        # Resize back to the original size
        denoised_img = resize_to_original(denoised_tensor)

        # Save the denoised image
        denoised_img = to_pil(denoised_img)
        denoised_img.save(output_path)

        print(f"Saved: {output_path}")

# Process train, valid, and test splits
for split in ["train", "valid", "test"]:
    print(f"Processing {split} data...")
    process_and_save_images(split)
