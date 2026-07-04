import torch
from torch import nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequential_thing = nn.Sequential(
            nn.Linear(3, 1)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.sequential_thing(x)
        return x

model = SimpleMLP()

X = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype=torch.float32)

logits = model(X)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.adamw(model.parameters(), lr = 0.001)
y = torch.tensor([1.0, 0.0])

def train(X, y, model,loss_fn,optimizer):
    model.train()
    for epoch in range(100):
        y_pred = model(X)
        loss = loss_fn(X,y_pred)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(X, y, model, loss_fn):
    model.eval()
    with torch.inference_mode():
        y_pred = model(X)
        loss = loss_fn(X,y_pred)
    return loss

def test2(model,X):
    with torch.inference_mode():
        y_pred = model(X)
        return y_pred