import torch.nn as nn
import torch
from Data.DataLoader import train_loader, test_loader
from Model.Vision_ import VisionTranformer
from train import train_model
from Config import config

model = VisionTranformer(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model(model=model, train_loader=train_loader, val_loader=test_loader, epochs=config["epochs"], device=device)