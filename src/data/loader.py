import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,   
                           saturation=0.2, hue=0.1),

     transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),

])                     

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
])

def get_data_loaders(train_dir, test_dir, device, batch_size=32):
    train_data = ImageFolder(root=train_dir, transform= train_transform)
    test_data = ImageFolder(root=test_dir, transform=test_transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                               generator=torch.Generator(device=device),
                              )
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False,
                             
                                   generator=torch.Generator(device=device),
                         )
    class_names = train_data.classes
    num_classes = len(class_names)
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {class_names}')
    return train_loader, test_loader, num_classes, class_names