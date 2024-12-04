import torch
import numpy as np
import torch.nn.functional as F

class GameOfLifeModel(torch.nn.Module):
    def __init__(self):
        super(GameOfLifeModel, self).__init__()
        self.conv = torch.Conv2D(1, 1, (3, 3), padding=1)
        self.fc = torch.nn.Linear(100, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x.view(-1, 100*100))
        x = F.tanh(x)
        return x.view(-1,1,100,100)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = GameOfLifeModel().to(device)

