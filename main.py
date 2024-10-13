import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import ConvNet
from utils import outer_loop, evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = load_data(train=True)

model = ConvNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    outer_loop(model, optimizer, loss_fn, [train_loader], device)

new_loader = load_data(train=False)

accuracy = evaluate_model(model, new_loader, device)

print(f"Accuracy on the new task: {accuracy:.2f}%")
