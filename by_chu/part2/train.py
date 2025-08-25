import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # torch import より前に必須

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import LunaModel
from dset import LunaDataset


def train(model: nn.Module, dataloader, batch_size=64, num_epoch=3,
          device="mps" if torch.backends.mps.is_available() else "cpu"):
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                          weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epoch + 1):
        running_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epoch}"):
            inputs, labels, _series, _centers = batch
            inputs = inputs.to(device)
            targets = labels[:, 1].to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}: loss{running_loss}")


if __name__ == "__main__":
    ds = LunaDataset(val_stride=10, isValSet_bool=False)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    model = LunaModel()
    train(model, dl, batch_size=256, num_epoch=1)
