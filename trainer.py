import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

class Trainer:
    def __init__(self, model, device, criterion, optimizer):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        # Initialize lists to store metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train(self, train_loader, valid_loader, num_epochs=10):
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
    
                self.optimizer.zero_grad()
    
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
    
                loss.backward()
                self.optimizer.step()
    
                running_loss += loss.item() * inputs.size(0)
    
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = 100 * correct / total
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
            # Validation phase
            val_loss, val_acc = self.evaluate(valid_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    
    def evaluate(self, data_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
    
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
    
                val_loss += loss.item() * inputs.size(0)
    
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        avg_loss = val_loss / len(data_loader.dataset)
        avg_acc = 100 * correct / total
        return avg_loss, avg_acc
    
    def test(self, test_loader):
        test_loss, test_acc = self.evaluate(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        return test_loss, test_acc
    
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
    
        # Plot Accuracy
        plt.figure(figsize=(10,5))
        plt.title("Training and Validation Accuracy")
        plt.plot(self.train_accuracies,label="Train Accuracy")
        plt.plot(self.val_accuracies,label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.show()

    def predict(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_preds, all_labels
    


    def plot_confusion_matrix(self, data_loader, class_names):
        all_preds, all_labels = self.predict(data_loader)
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8,6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', xticklabels=class_names,
                    yticklabels=class_names, cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Normalized Confusion Matrix')
        plt.show()


    def visualize_predictions(self, data_loader, class_names, num_images=8):
        self.model.eval()
        inputs, labels = next(iter(data_loader))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)

        inputs = inputs.cpu()
        preds = preds.cpu()
        labels = labels.cpu()

        def imshow(inp, title=None):
            inp = inp.numpy().transpose((1, 2, 0))
            mean = [0.485, 0.456, 0.406]
            std =[0.229, 0.224, 0.225]
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            if title is not None:
                plt.title(title)
            plt.axis('off')

        fig = plt.figure(figsize=(15, 10))
        for idx in np.arange(num_images):
            ax = fig.add_subplot(2, num_images//2, idx+1)
            imshow(inputs[idx])
            ax.set_title(f"Pred: {class_names[preds[idx]]}\nTrue: {class_names[labels[idx]]}",
                        color=("green" if preds[idx]==labels[idx] else "red"))
        plt.show()
