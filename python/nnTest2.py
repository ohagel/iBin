import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from trashDataset import TrashDataset
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    batch_size = 1
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = TrashDataset(csvFile='dataset/labels.txt', root_dir='dataset', transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 117 * 117, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 4)

        def forward(self, img, weight):
            x = self.pool(F.relu(self.conv1(img)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            imgs, weight, labels = data['image'].to(device), data['weight'].to(device), data['class'].to(device)
            #imgs, weight, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            print('THIS IS IT INPUTS')
            print(imgs.shape)
            print('THIS IS IT WEIGHTS')
            print(weight)
            print('THIS IS IT LABELS')
            print(labels)

            optimizer.zero_grad()
            outputs = net(imgs, weight)
            print('THIS IS IT OUTPUTS')
            print(outputs)
            quit()
            #loss = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()