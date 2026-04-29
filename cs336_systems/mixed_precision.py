import torch
import torch.nn as nn
from torch import autocast

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    x = torch.randn(10, 100)
    dtype = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ToyModel(100, 100)
    opt = torch.optim.Adam(model.parameters())

    with autocast(device_type=device, dtype=torch.float16):
        opt.zero_grad()
        y: torch.Tensor = model(x)
        loss = y.sum()
        loss.backward()
        opt.step()

    x, loss, y



# if __name__ == '__main__':
#     s = torch.tensor(0,dtype=torch.float32)
#     for i in range(1000):
#         s += torch.tensor(0.01,dtype=torch.float32)
#     print(s)

#     s = torch.tensor(0,dtype=torch.float16)
#     for i in range(1000):
#         s += torch.tensor(0.01,dtype=torch.float16)
#     print(s)

#     s = torch.tensor(0,dtype=torch.float32)
#     for i in range(1000):
#         s += torch.tensor(0.01,dtype=torch.float16)
#     print(s)

#     s = torch.tensor(0,dtype=torch.float32)
#     for i in range(1000):
#         x = torch.tensor(0.01,dtype=torch.float16)
#         s += x.type(torch.float32)
#     print(s)