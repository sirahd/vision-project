import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels
from torch.autograd import Variable

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        if not os.path.exists('logs'):
            os.makedirs('logs')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open('logs/' + st, 'w')

    def log(self, str):
        print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        return nn.BCEWithLogitsLoss()

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.lr  # TODO: Implement decreasing learning rate's rules
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class LazyNet(BaseModel):
    def __init__(self):
        super(LazyNet, self).__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(256 * 256 * 3, 1)

    def forward(self, x):
        # TODO: Implement forward pass for LazyNet
        x = self.fc1(x.view(-1, 256 * 256 * 3))
        return x

class BoringNet(BaseModel):
    def __init__(self):
        super(BoringNet, self).__init__()
        # TODO: Define model here
        self.fc1 = nn.Linear(32 * 32 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO: Implement forward pass for BoringNet
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CoolNet(BaseModel):
    def __init__(self):
        super(CoolNet, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, 11, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3136, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)


    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        a = F.max_pool2d(F.relu(self.conv1(x)), 2)
        b = F.max_pool2d(F.relu(self.conv2(a)), 2)
        c = b.view(-1, num_flat_features(b))
        d = F.relu(self.fc1(c))
        e = F.relu(self.fc2(d))
        f = self.fc3(e)
        return f

class VGG(BaseModel):
    def __init__(self):
        super(VGG, self).__init__()
        # TODO: Define model here
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
        )
        self.features = self.make_layers(True)

    def make_layers(self, batch_norm=False):
        layers = []
        in_channels = 3
        for v in [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M']:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
