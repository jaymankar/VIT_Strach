import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from Data.DataLoader import train_loader, test_loader
from Model.Vision_ import VisionTranformer
from train import train_model
from Config import config

model = VisionTranformer(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_and_show(model, loader, device, num_images=10):

    model.eval()

    images, labels = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

    images = images.cpu()

    plt.figure(figsize=(12,5))

    for i in range(num_images):

        plt.subplot(2,5,i+1)

        img = images[i].squeeze()

        plt.imshow(img, cmap="gray")

        pred = preds[i].item()
        true = labels[i].item()

        plt.title(f"Pred: {pred} | True: {true}")

        plt.axis("off")

    plt.tight_layout()
    plt.show()



predict_and_show(model=model, loader=test_loader, device=device)    import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from Data.DataLoader import train_loader, test_loader
from Model.Vision_ import VisionTranformer
from train import train_model
from Config import config

model = VisionTranformer(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_and_show(model, loader, device, num_images=10):

    model.eval()

    images, labels = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

    images = images.cpu()

    plt.figure(figsize=(12,5))

    for i in range(num_images):

        plt.subplot(2,5,i+1)

        img = images[i].squeeze()

        plt.imshow(img, cmap="gray")

        pred = preds[i].item()
        true = labels[i].item()

        plt.title(f"Pred: {pred} | True: {true}")

        plt.axis("off")

    plt.tight_layout()
    plt.show()



predict_and_show(model=model, loader=test_loader, device=device)    