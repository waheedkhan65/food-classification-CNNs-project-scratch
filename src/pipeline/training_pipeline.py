"""
    Script to run the training pipeline for the model.
    :param file_path: Path to the dataset CSV file.
    :param target_column: Name of the target column.
    """

import torch.nn as nn
import torch
import torch.optim as optim
from src.models.cnn import CNN
from src.data.loader import get_data_loaders
import matplotlib.pyplot as plt

def training_loop(train_loader, model, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch,(image, label) in enumerate(train_loader):
        image = image.to(torch.device(device))
        label = label.to(torch.device(device))
        
        optimizer.zero_grad()
        
        # Forward Pass
        output = model(image)
        
        loss = criterion(output, label)
        
        # Backward Pass
        loss.backward()
        
        optimizer.step()


        running_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        
        if batch % 5 == 0:
            print(f'Batch {batch}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validation_loop(test_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, (image, label) in enumerate(test_loader):
            image = image.to(torch.device(device))
            label = label.to(torch.device(device))
            
            output = model(image)
            loss = criterion(output, label)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            if batch % 5 == 0:
                print(f'Batch {batch}/{len(test_loader)}, Loss: {loss.item():.4f}')
    
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def epoch_loop(EPOCHS, model, train_loader, test_loader, optimizer, criterion, device):
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    best_val_acc = 0.0
    for epoch in range (EPOCHS):
        training_loss, training_acc = training_loop(train_loader, model, optimizer, criterion, device)
        validation_loss, validation_acc = validation_loop(test_loader, model, criterion, device)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        training_accuracies.append(training_acc)
        validation_accuracies.append(validation_acc)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {training_loss:.4f}, Accuracy: {training_acc:.2f}%')

        print(f'Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_acc:.2f}%')

        if validation_acc > best_val_acc:
            best_val_acc = validation_acc
            print(f'New best validation accuracy: {best_val_acc:.2f}%')
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    print("Training complete.")
    return training_losses, validation_losses, training_accuracies, validation_accuracies

def main():
    # You can change the number of epochs as needed
    epochs = 120
    

    train_dir = "data/processed/train"
    test_dir = "data/processed/test"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", DEVICE)
    torch.set_default_device(DEVICE)
    train_loader, test_loader, num_classes, class_names = get_data_loaders(train_dir, test_dir,DEVICE,batch_size=32)
    model = CNN(num_classes)  # Adjust num_classes as needed
    LEARNING_RATE = 0.001
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    training_losses, validation_losses, training_accuracies, validation_accuracies = epoch_loop(epochs, model, train_loader, test_loader, optimizer, criterion, DEVICE)
    # Plotting the training and validation losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()    

    plt.subplot(1, 2, 2)
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
