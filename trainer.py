import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc, multilabel_confusion_matrix
import matplotlib.patches as patches

class Trainer:
    def __init__(self, model, device, criterion, optimizer, num_classes=7, class_names=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None else [f'Class {i}' for i in range(num_classes)]
        # Initialize lists to store metrics
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, valid_loader, num_epochs=10):
        for epoch in range(num_epochs):
            # Training phase
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
    
            # Validation phase
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
    
        # Compute Average Precision (AP) for each class
        average_precisions = []
        for i in range(self.num_classes):
            ap = average_precision_score(all_targets[:, i], all_outputs[:, i])
            average_precisions.append(ap)
            print(f'Class {self.class_names[i]} AP: {ap:.4f}')
    
        # Compute mean Average Precision (mAP)
        mAP = np.mean(average_precisions)
        print(f'mAP: {mAP:.4f}')
    
        return all_targets, all_outputs, average_precisions, mAP
    
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
    
    def plot_roc_curves(self, targets, outputs):
        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(targets[:, i], outputs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
    
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()
    
    def plot_confusion_matrices(self, targets, outputs, threshold=0.5):
        preds = (outputs >= threshold).astype(int)
        confusion_matrices = multilabel_confusion_matrix(targets, preds)
    
        for i in range(self.num_classes):
            cm = confusion_matrices[i]
            print(f'Confusion matrix for class {self.class_names[i]}:')
            print(cm)
            print()
    
    def visualize_predictions(self, data_loader, num_images=8, threshold=0.5):
        import numpy as np
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
    
        # Denormalize images
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
    
            # Get bounding boxes for this image
            bboxes = bboxes_list[idx]
    
            # Draw bounding boxes
            for bbox in bboxes:
                xmin, ymin, xmax, ymax, class_id = bbox
                width = xmax - xmin
                height = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2,
                                         edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                # Add class label
                ax.text(xmin, ymin - 5, self.class_names[class_id], color='red', fontsize=12,
                        backgroundcolor='white')
        plt.tight_layout()
        plt.show()
