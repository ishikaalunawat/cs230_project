import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    def __init__(self, model, dataloaders, optimizer, loss_fn, device, logdir="runs"):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.device = device
        # for tensorboard
        self.writer = SummaryWriter(log_dir=logdir)

    @staticmethod
    def calculate_psnr(output, target, max_pixel_value=1.0):
        # Ensure the inputs are on CPU and convert to float
        output = output.detach().cpu()
        target = target.detach().cpu()
        
        # Calculate MSE across all dimensions except batch
        mse = torch.mean((output - target) ** 2, dim=[1,2,3])
        
        # Handle zero MSE case
        zero_mask = (mse == 0)
        mse[zero_mask] = torch.finfo(torch.float32).eps  # Small epsilon instead of 0
        
        # Calculate PSNR
        psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
        
        # Set infinity for zero MSE cases
        psnr[zero_mask] = float('inf')
        
        # Return mean PSNR across batch
        return psnr.mean().item()

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        total_psnr = 0.0
        progress_bar = tqdm(self.dataloaders['train'], desc=f"Training Epoch {epoch+1}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            noisy_images = batch["noisy"].to(self.device)
            clean_images = batch["clean"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(noisy_images)
            loss = self.loss_fn(outputs, clean_images)
            loss.backward()
            self.optimizer.step()

            # PSNR
            psnr = self.calculate_psnr(outputs, clean_images)
            total_psnr += psnr

            # update running loss
            batch_loss = loss.item()
            running_loss += batch_loss

            # log running loss and PSNR to TensorBoard
            global_step = epoch * len(self.dataloaders['train']) + batch_idx
            self.writer.add_scalar("Loss/Train_Batch", batch_loss, global_step)
            self.writer.add_scalar("PSNR/Train_Batch", psnr, global_step)

            # fro tqdm
            progress_bar.set_postfix(Batch_Loss=batch_loss, Batch_PSNR=psnr)

        # log epoch
        epoch_loss = running_loss / len(self.dataloaders['train'])
        epoch_psnr = total_psnr / len(self.dataloaders['train'])
        self.writer.add_scalar("Loss/Train_Epoch", epoch_loss, epoch)
        self.writer.add_scalar("PSNR/Train_Epoch", epoch_psnr, epoch)

        return epoch_loss, epoch_psnr

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        total_psnr = 0.0
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['valid'], desc="Validating", leave=False):
                noisy_images = batch["noisy"].to(self.device)
                clean_images = batch["clean"].to(self.device)

                outputs = self.model(noisy_images)
                loss = self.loss_fn(outputs, clean_images)

                # PSNR
                psnr = self.calculate_psnr(outputs, clean_images)
                total_psnr += psnr

                running_loss += loss.item()

        epoch_loss = running_loss / len(self.dataloaders['valid'])
        epoch_psnr = total_psnr / len(self.dataloaders['valid'])
        self.writer.add_scalar("Loss/Validation", epoch_loss, epoch)
        self.writer.add_scalar("PSNR/Validation", epoch_psnr, epoch)

        return epoch_loss, epoch_psnr

    def visualize_test_outputs(self):
        self.model.eval()
        test_loader = self.dataloaders.get("test")
        if not test_loader:
            print("Test loader not loaded.")
            return

        test_iter = iter(test_loader)
        batch = next(test_iter)

        noisy_images = batch["noisy"].to(self.device)
        clean_images = batch["clean"].to(self.device)

        with torch.no_grad():
            outputs = self.model(noisy_images)

        # convert to correct shape
        noisy_images = noisy_images.cpu().numpy().transpose(0, 2, 3, 1)
        outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)
        clean_images = clean_images.cpu().numpy().transpose(0, 2, 3, 1)

        # denormalize
        noisy_images = np.clip(noisy_images, 0, 1)
        outputs = np.clip(outputs, 0, 1)
        clean_images = np.clip(clean_images, 0, 1)

        # plot
        fig, axes = plt.subplots(5, 3, figsize=(12, 15))
        for i in range(5):
            axes[i, 0].imshow(noisy_images[i])
            axes[i, 0].set_title("Input (Noisy)")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(outputs[i])
            axes[i, 1].set_title("Model Output")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(clean_images[i])
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig("outputs/test_visualization.png")
        plt.show()

    def close(self):
        # to close tensorboard writer
        self.writer.close()