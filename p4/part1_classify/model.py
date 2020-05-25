import torch.nn as nn
import torch.nn.functional as F
import torch
# import tensorwatch as tw
import torchvision.models
# import hiddenlayer as hl


class Class_Net(nn.Module):
    def __init__(self):
        super(Class_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 4 * 4, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x

class Class_Net96(nn.Module):
    def __init__(self):
        super(Class_Net96, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(96, 96, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(96 * 4 * 4, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 96 * 4 * 4)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x

if __name__=="__main__":
    model = Class_Net()
    # tw.draw_model(model, [1, 1, 28, 28])
    # drawing.save('model.png')
    alexnet_model = torchvision.models.alexnet()
    hl.build_graph(alexnet_model, torch.zeros([1, 3, 224, 224]))
    # drawing = tw.draw_model(alexnet_model, [1, 3, 224, 224], png_filename='model.png')
    # drawing.save('model.png')

