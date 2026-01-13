from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,2)

    def forward(self, x):
        x = self.l1(x)
        return x
