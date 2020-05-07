from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch
from Squeezenet import SqueezeNet

torch.manual_seed(123)
train_path = 'train_path'
test_path = 'test_path'

train_data_loader = DataLoader(
    torchvision.datasets.ImageFolder(
        train_path,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # INPUT IMAGE SIZE FOR SQUEEZENET
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10),
            transforms.RandomPerspective(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ),
    num_workers=8,
    batch_size=128,
    shuffle=True,
    pin_memory=False
)
'''
test_data_loader = DataLoader(
    torchvision.datasets.ImageFolder(
        train_path,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10),
            transforms.RandomPerspective(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #suggested values for normalization
        ])
    ),
    num_workers=8,
    batch_size=32,
    shuffle=True,
    pin_memory=False
)
'''
squeezeNet = SqueezeNet(channels=3, classes=29)
squeezeNet.train_model(epochs=50, data=train_data_loader, model=squeezeNet)
