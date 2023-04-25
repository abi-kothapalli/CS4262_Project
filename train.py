import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import numpy as np
from datetime import datetime
from collections import defaultdict
import json
import argparse

from data_utils import get_dataloaders

DEVICE = "cuda:1"
AVAILABLE_DEVICES = [1, 2, 3]
DEFAULT_EPOCHS = 30


def get_model(name = "resnet"):
    if name not in ["resnet", "vit"]:
        raise ValueError(f"Model {name} not supported")
    
    if name == "resnet":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
    else:
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.heads = torch.nn.Linear(in_features=768, out_features=1, bias=True)

    model = nn.DataParallel(model, device_ids=AVAILABLE_DEVICES).to(DEVICE)
    return model

def train_model(model_name, epochs = DEFAULT_EPOCHS):
    
    model = get_model(model_name)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loader, test_loader = get_dataloaders()

    metrics = defaultdict(list)
    curr_time = datetime.now().strftime("%Y-%m-%d")
    writer = SummaryWriter(log_dir=f"./runs/{model_name}/{curr_time}")


    for epoch in tqdm(range(epochs), desc="Epochs", position=0):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = test_loader
            
            loss_ = []
            acc_ = []

            for i, data in tqdm(enumerate(loader), desc=f"{phase.capitalize()}", position=1, leave=False, total=len(loader)):

                optimizer.zero_grad()

                x, y = data
                x = x.to(DEVICE)
                y = y.to(DEVICE).type(torch.float)

                with torch.inference_mode(mode=(phase == 'test')):
                    logits = model(x)
                    loss = loss_fn(logits.squeeze(), y)
                    preds = torch.sigmoid(logits).squeeze() > 0.5
                    acc = (1. * (preds == y)).mean().item()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                loss_.append(loss.item())
                acc_.append(acc)
            
            metrics[f"{phase}_loss"].append(np.mean(loss_))
            metrics[f"{phase}_acc"].append(np.mean(acc_))

            writer.add_scalar(f"{phase}_loss", np.mean(loss_), epoch)
            writer.add_scalar(f"{phase}_acc", np.mean(acc_), epoch)

        scheduler.step()
    
    with open(f"./runs/{model_name}/{curr_time}/metrics.json", "w") as f:
        json.dump(metrics, f)

    writer.close()
    return metrics

if __name__ == "__main__":\

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="resnet", help="Model to train")
    args = parser.parse_args()

    train_model(args.model)