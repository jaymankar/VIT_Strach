import torch.nn as nn
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
from Config import config

train_dataset = datasets.MNIST(root = "./dataset", train=True, download=True , transform=torchvision.transforms.ToTensor())
test_dataset = datasets.MNIST(root = "./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=config["batch_size"], shuffle=True)




