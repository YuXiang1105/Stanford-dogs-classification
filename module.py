from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,2)
        self.l2 = nn.Linear(2,1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x
