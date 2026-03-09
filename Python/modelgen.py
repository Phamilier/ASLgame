import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class ASLmodel(nn.Module):
    def __init__(self, num_classes=24):
        super(ASLmodel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(F.relu(self.bn3(self.conv3(x))))

        x = self.global_pool(x)  # (batch, 256, 8, 8) → (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256, 1, 1) → (batch, 256)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def load_data():
    x_train = np.load('x_train.npy')
    x_test = np.load('x_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    if len(y_train.shape) > 1 and y_train.shape[1] == 24:
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

    x_train = torch.FloatTensor(x_train).permute(0, 3, 1, 2)
    x_test = torch.FloatTensor(x_test).permute(0, 3, 1, 2)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")

    return x_train, x_test, y_train, y_test

def train_model(model, train_loader, test_loader, epochs = 20, lr = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y= batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        train_accuracy = train_correct / train_total
        test_accuracy = test_correct / test_total

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)

        print(f"Epoch: {epoch+1}/{epochs}"
              f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_accuracy:.4f} "
              f"Test Loss: {test_loss:.4f} "  
              f"Test Acc: {test_accuracy:.4f}")

    return train_losses, train_accuracies, test_losses, test_accuracies

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    sign_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
                  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize = (12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap = 'Blues', xticklabels=sign_names, yticklabels=sign_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def export_to_onnx(model, filepath = 'asl.onnx_pytorch'):
    model.eval()
    dummy_input = torch.randn(1, 1, 32, 32)

    torch.onnx.export(model, dummy_input, filepath)

def plot_training_history(training_losses, training_accuracies, test_losses, test_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))

    ax1.plot(training_losses, label = 'Training Loss')
    ax1.plot(test_losses, label = 'Test Loss')
    ax1.title.set_text('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(training_accuracies, label = 'Training Accuracy')
    ax2.plot(test_accuracies, label = 'Test Accuracy')
    ax2.title.set_text('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.show()

def main():
    x_train, x_test, y_train, y_test = load_data()

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False)

    model = ASLmodel(num_classes = 24)

    train_losses, train_accuracies, test_losses, test_accuracies = train_model(model, train_loader, val_loader, epochs = 20, lr = 0.001)

    plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies)
    evaluate_model(model, val_loader)

    export_to_onnx(model)

if __name__ == '__main__':
    main()