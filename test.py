import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_1 = nn.Linear(4, 4)
        self.linear_2 = nn.Linear(4, 4)
        self.linear_3 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = F.relu(x)
        return x


if __name__ == '__main__':
    net = Net().to(torch.device('cuda:0'))
    for parameter in net.parameters():
        print(parameter.device)
