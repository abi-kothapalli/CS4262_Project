import torch
from torch import nn
import torchvision
import numpy as np
from tqdm import tqdm
import timeit

from data_utils import get_dataloaders

start = timeit.default_timer()

DEVICE = "cuda:1"
AVAILABLE_DEVICES = [1]

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights)
model.fc = Identity()
model = nn.DataParallel(model, device_ids=AVAILABLE_DEVICES).to(DEVICE)

train_loader, test_loader = get_dataloaders()

model.eval()
with torch.inference_mode():
    for name, loader in zip(["train", "test"], [train_loader, test_loader]):
        final_matrix = None

        for x, y in tqdm(loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE).type(torch.float)

            feats = model(x)

            total = torch.hstack((feats, y.unsqueeze(1)))
            final_matrix = total if final_matrix is None else torch.vstack((final_matrix, total))

            np.save(f"{name}_data.npy", final_matrix.cpu().numpy())


print(f"Time taken: {timeit.default_timer() - start} seconds")