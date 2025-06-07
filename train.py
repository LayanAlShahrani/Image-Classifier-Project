# programmer : Layan
# Date : 2/1/2024

import argparse
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    # Define transforms for training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(f'{data_dir}/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(f'{data_dir}/valid', transform=valid_transforms)

    # Using DataLoader to load data in batches
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=32)

    return trainloader, validloader, train_data.class_to_idx

def train_model(trainloader, validloader, class_to_idx, lr=0.001, epochs=10):
    # Load a pre-trained network (VGG16 in this example)
    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    # Set the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    # Train the classifier layers using backpropagation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 50

    for epoch in range(epochs):
        # Training loop
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Validation loop
                validation_loss = 0
                accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, labels)
                        validation_loss += batch_loss.item()
                        
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                
                running_loss = 0
                model.train()

    # Save the checkpoint
    model.class_to_idx = class_to_idx
    model.to('cpu')
    checkpoint = {'class_to_idx': model.class_to_idx, 'model_state_dict': model.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a flower classification model')
    parser.add_argument('--data_dir', type=str, default='flower_data', help='Path to the dataset directory')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    args = parser.parse_args()

    trainloader, validloader, class_to_idx = load_data(args.data_dir)
    train_model(trainloader, validloader, class_to_idx, lr=args.learning_rate, epochs=args.epochs)
