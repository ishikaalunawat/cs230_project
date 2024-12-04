import os
import torch
from PIL import Image
from torchvision import transforms
from models_inspiration import MAD 

# configs
data_dir = "datasets_noisy/aquarium-data-cots/aquarium_pretrain"
model_path = "outputs_/denoiser_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transforms
resize_to_model = transforms.Compose([
    transforms.Resize((256, 256)),  # for the model
    transforms.ToTensor()
])
resize_to_original = transforms.Compose([
    transforms.Resize((1024, 768))  # back to original size
])
to_pil = transforms.ToPILImage()

# load saved model
model = MAD(in_channels=3, out_channels=3, num_features=64, num_blocks=8).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def process_and_save_images(data_split):
    input_folder = os.path.join(data_dir, data_split, "noisy_images")
    output_folder = os.path.join(data_dir, data_split, "denoised_imgs")

    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        img = Image.open(input_path).convert("RGB")

        # preprocess
        img_tensor = resize_to_model(img).unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            denoised_tensor = model(img_tensor).squeeze(0).cpu()

        # resize
        denoised_img = resize_to_original(denoised_tensor)

        # save
        denoised_img = to_pil(denoised_img)
        denoised_img.save(output_path)

        print(f"Saved: {output_path}")

# process for all
for split in ["train", "valid", "test"]:
    print(f"Processing {split} data...")
    process_and_save_images(split)
