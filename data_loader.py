import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

def load_data(train=True, batch_size=64):
    dataset = MNIST(root='data/', train=train, transform=ToTensor(), download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
