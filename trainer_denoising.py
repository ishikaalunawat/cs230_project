import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.patches as patches
from collections import Counter
import seaborn as sns
#DenoisingTrainer:
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr #DenoisingTrainer
from skimage.metrics import structural_similarity as ssim #DenoisingTrainer
from torchvision.models import vgg16
# from pytorch_msssim import ssim
import torch.optim as optim

def perceptual_loss(output, target):
    ''' Perceptual loss calculation function for Denoising model.
    '''
    vgg = vgg16(pretrained=True).features[:16]
    vgg.eval()
    output_features = vgg(output)
    target_features = vgg(target)
    return torch.mean((output_features - target_features) ** 2)


def combined_loss(output, target):
    ''' Combined loss function with MSE, perceptual loss, and SSIM loss for Denoising model.
    '''
    perceptual = perceptual_loss(output, target)
    
    mse_loss = nn.MSELoss()(output, target)
    
    ssim_loss = 1 - ssim(output, target, data_range=1.0, size_average=True)

    return mse_loss + 0.1 * ssim_loss + 0.5 * perceptual

class Trainer:
    def __init__(self, model, device, criterion, optimizer, num_classes=7, class_names=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None else [f'Class {i}' for i in range(num_classes)]
        # to track & plot losses
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, valid_loader, num_epochs=10):
        for epoch in range(num_epochs):
            # train
            self.model.train()
            running_loss = 0.0
            total_samples = 0
            for inputs, labels, _ in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()
    
                self.optimizer.zero_grad()
    
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
    
                loss.backward()
                self.optimizer.step()
    
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
    
            epoch_loss = running_loss / total_samples
            self.train_losses.append(epoch_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
    
            # eval
            val_loss = self.evaluate_loss(valid_loader)
            self.val_losses.append(val_loss)
            print(f'Validation Loss: {val_loss:.4f}')
    
    def evaluate_loss(self, data_loader):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels, _ in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()
    
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
    
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
    
        avg_loss = running_loss / total_samples
        return avg_loss
    
    def evaluate_metrics(self, data_loader):
        self.model.eval()
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for inputs, labels, _ in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()
    
                outputs = self.model(inputs)
                outputs = torch.sigmoid(outputs)
    
                all_targets.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
    
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)
    
        # get avg. precision for each class
        average_precisions = []
        for i in range(self.num_classes):
            ap = average_precision_score(all_targets[:, i], all_outputs[:, i])
            average_precisions.append(ap)
            print(f'Class {self.class_names[i]} AP: {ap:.4f}')
    
        # get map
        map = np.mean(average_precisions)
        print(f'mAP: {map:.4f}')
    
        return all_targets, all_outputs, average_precisions, map
    
    def plot_metrics(self):
        # Plot Loss
        plt.figure(figsize=(10,5))
        plt.title("Training and Validation Loss")
        plt.plot(self.train_losses,label="Train Loss")
        plt.plot(self.val_losses,label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    
    def compute_confusion_matrix(self, targets, outputs, threshold=0.5, normalize=True):
        num_classes = self.num_classes
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        
        preds = (outputs >= threshold).astype(int)

        for i in range(num_classes):
            for j in range(num_classes):
                # pairwise ij
                count = np.sum((targets[:, i] == 1) & (preds[:, j] == 1))
                confusion_matrix[i, j] = count

        if normalize:
            # div zero error
            row_sums = confusion_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            normalized_confusion = confusion_matrix / row_sums
            return normalized_confusion
        else:
            return confusion_matrix

    

    def plot_pairwise_confusion_matrix(self, confusion_matrix, class_names, epoch, threshold=0.5):
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix.T, annot=True, fmt=".2f", cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'Confusion Matrix Normalized')
        plt.tight_layout()
        plt.show()


    
    def visualize_predictions(self, data_loader, num_images=8, threshold=0.5):
        self.model.eval()
        inputs, labels, bboxes_list = next(iter(data_loader))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device).float()
        with torch.no_grad():
            outputs = self.model(inputs)
            outputs = torch.sigmoid(outputs)
        inputs = inputs.cpu()
        labels = labels.cpu()
        outputs = outputs.cpu()
    
        preds = (outputs >= threshold).int()
    
        # denorm images for viz
        def denormalize(inp):
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std =[0.229, 0.224, 0.225]
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            return inp
    
        fig = plt.figure(figsize=(15, 10))
        for idx in range(num_images):
            ax = fig.add_subplot(2, num_images//2, idx+1)
            image = denormalize(inputs[idx])
            ax.imshow(image)
            true_classes = [self.class_names[i] for i in range(self.num_classes) if labels[idx][i] == 1]
            pred_classes = [self.class_names[i] for i in range(self.num_classes) if preds[idx][i] == 1]
            title = f"True: {', '.join(true_classes)}\nPred: {', '.join(pred_classes)}"
            ax.set_title(title)
            ax.axis('off')
    
            
            bboxes = bboxes_list[idx]
    
            # to draw bbox
            for bbox in bboxes:
                xmin, ymin, xmax, ymax, class_id = bbox
                width = xmax - xmin
                height = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2,
                                         edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                # class label
                ax.text(xmin, ymin - 5, self.class_names[int(class_id)], color='red', fontsize=12,
                        backgroundcolor='white')
        plt.tight_layout()
        plt.show()


class DenoisingTrainer:
    def __init__(self, model, device, criterion, optimizer, save_path='best_model.pth'):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_path = save_path

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train(self, train_loader, valid_loader, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            running_train_loss = 0.0
            total_train_samples = 0

            for noisy_images, clean_images in train_loader:
                noisy_images = noisy_images.to(self.device)
                clean_images = clean_images.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(noisy_images)
                loss = self.criterion(outputs, clean_images) #loss definition here -- depends on definition in training pipeline
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item() * noisy_images.size(0)
                total_train_samples += noisy_images.size(0)

            epoch_train_loss = running_train_loss / total_train_samples
            self.train_losses.append(epoch_train_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}')

            self.model.eval()
            running_val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for noisy_images, clean_images in valid_loader:
                    noisy_images = noisy_images.to(self.device)
                    clean_images = clean_images.to(self.device)

                    outputs = self.model(noisy_images)
                    loss = self.criterion(outputs, clean_images) #Criterion referenced here

                    running_val_loss += loss.item() * noisy_images.size(0)
                    total_val_samples += noisy_images.size(0)

            epoch_val_loss = running_val_loss / total_val_samples
            self.val_losses.append(epoch_val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}')

            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Model saved with validation loss: {epoch_val_loss:.4f}")

        #Plotting train and validation loss per epoch at the end of train() function
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate(self, data_loader):
        ''' Evaluates MSE loss, pSNR, and SSIM of the model output to the ground truth labels
            using skimage.metrics for the latter two metrics.
        '''
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        total_psnr = 0.0
        total_ssim = 0.0

        with torch.no_grad():
            for noisy_images, clean_images in data_loader:
                noisy_images = noisy_images.to(self.device)
                clean_images = clean_images.to(self.device)

                outputs = self.model(noisy_images)
                loss = self.criterion(outputs, clean_images) #Criterion

                running_loss += loss.item() * noisy_images.size(0)
                total_samples += noisy_images.size(0)

                # METRICS pSNR, SSIM ADDED HERE
                outputs_np = outputs.cpu().numpy().transpose(0, 2, 3, 1)
                clean_images_np = clean_images.cpu().numpy().transpose(0, 2, 3, 1)
                
                # debugging ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                # print("Output shape:", outputs_np.shape)
                # print("Clean image shape:", clean_images_np.shape)
                # end debugging ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

                for o, c in zip(outputs_np, clean_images_np):
                    total_psnr += psnr(c, o, data_range=1.0)
                    total_ssim += ssim(c, o, data_range=1.0, win_size=3, channel_axis=-1)

        avg_loss = running_loss / total_samples
        avg_psnr = total_psnr / total_samples
        avg_ssim = total_ssim / total_samples

        print(f"Avg Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}")
        return avg_loss, avg_psnr, avg_ssim
