import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# print(torch.backends.mps.is_built() )
# print(torch.backends.mps.is_available())

# print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

print(logits)
print(pred_probab)

print(f"Predicted class: {y_pred}")

print("------------------------------------")
# Self test
import numpy as np

ndArray = np.array([[1,2,3],[4,5,6],[7,8,9]])

tensor = torch.tensor(ndArray, dtype=float)

# 열의 합이 1
print(nn.Softmax(dim=0)(tensor))

# 행의 합이 1
print(nn.Softmax(dim=1)(tensor))