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
from torch.utils.data import DataLoader, SubsetRandomSampler
import cv2
import datetime

class Net(nn.Module):
        
        def __init__(self, csv_file=None, rootdir=None, split=0.2, device = None):
            super().__init__()

            conv1_kernel_size=5
            conv2_kernel_size=3
            pool_kernel_size=10
            conv_stride=1
            pool_stride=10
            input_size=480

            seed = 42
            torch.manual_seed(seed)
            np.random.seed(seed)

            batch_size = 8
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.0494, 0.2506, 0.7143], std=[0.1260, 0.1261, 0.1255])])
            #self.transform = transforms.Compose([
            #    transforms.ToTensor(),
            #    transforms.RandomHorizontalFlip(),
            #    transforms.RandomVerticalFlip(),
            #    transforms.RandomRotation(45),
            #    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            #    transforms.Normalize(mean=[0.0494, 0.2506, 0.7143], std=[0.1260, 0.1261, 0.1255])
            #])

            self.csv_file = csv_file
            self.rootdir = rootdir

            if self.csv_file and self.rootdir:
                trainset = TrashDataset(csvFile=csv_file, root_dir=rootdir, transform=self.transform)

                # Define the split ratio for validation set
                validation_split = split
                dataset_size = len(trainset)
                indices = list(range(dataset_size))
                split = int(np.floor(validation_split * dataset_size))

                np.random.shuffle(indices)

                # Split indices into training and validation sets
                train_indices, val_indices = indices[split:], indices[:split]

                # Define samplers for obtaining batches from train and validation sets
                train_sampler = SubsetRandomSampler(train_indices)
                valid_sampler = SubsetRandomSampler(val_indices)

                #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                # Create data loaders for train and validation sets using the samplers
                self.train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
                self.validation_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,num_workers=2)

            if device:
                self.device = device
            else: 
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            self.to(self.device)
            print("Running on", self.device)
           

            self.conv1 = nn.Conv2d(3, 6, conv1_kernel_size)
            self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
            self.conv2 = nn.Conv2d(6, 16, conv2_kernel_size)

            #update input after conv1
            self.fc1_input_size = self.calc_input_size(input_size, conv1_kernel_size, conv_stride)
            #update input after pool1
            self.fc1_input_size = self.calc_input_size(self.fc1_input_size, pool_kernel_size, pool_stride)
            #update input after conv2
            self.fc1_input_size = self.calc_input_size(self.fc1_input_size, conv2_kernel_size, conv_stride)
            #update input after pool2
            self.fc1_input_size = self.calc_input_size(self.fc1_input_size, pool_kernel_size, pool_stride)
        
            self.fc1_input_size = self.fc1_input_size**2*16
            self.fc1_input_size = int(self.fc1_input_size)+1
            
            self.fc1 = nn.Linear(self.fc1_input_size, 120)

            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 4)

        def forward(self, img, weight):
            x = self.pool(F.relu(self.conv1(img)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            weight = weight.view(-1, 1)
            x = torch.cat((x, weight), dim=1) #input weight to fc1
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
        def calc_input_size(self, input_size, kerne_size, stride):
            return (input_size - kerne_size)//stride+1
        
        def load(self, model_path):
            self.load_state_dict(torch.load(model_path))
            self.to(self.device)
        
        def train(self, n_epochs, save_path=None):
            if self.csv_file and self.rootdir:
                criterion = nn.CrossEntropyLoss()
                self.to(self.device)
                #optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
                optimizer = optim.AdamW(self.parameters(), lr=0.001)

                for epoch in range(n_epochs):  # loop over the dataset multiple times

                    running_loss = 0.0
                    for i, data in enumerate(self.train_loader, 0):

                        imgs, weight, labels = data['image'].to(self.device), data['weight'].to(self.device), data['class'].to(self.device)
                        optimizer.zero_grad()
                        outputs = self(imgs, weight)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        

                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.10f}')
                        
                print('Finished Training')
                if save_path:
                    PATH = save_path
                    print("saving model to", PATH, "...")
                else:
                    ##save model with timestamp as dd/mm/YY H:M
                    now = datetime.datetime.now()
                    PATH = './iBin_net2_' + now.strftime("%d-%m-%Y_%H;%M") + '.pth'
                    print("saving model to", PATH, "...")
                torch.save(self.state_dict(), PATH)
                
            else:
                print("No dataset loaded")

        def validate(self):
            if self.csv_file and self.rootdir:
                correct = 0
                total = 0
                # since we're not training, we don't need to calculate the gradients for our outputs
                with torch.no_grad():
                    for data in self.validation_loader:

                        imgs, weight, labels = data['image'].to(self.device), data['weight'].to(self.device), data['class'].to(self.device)
                        # calculate outputs by running images through the network
                        outputs = self(imgs, weight)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                print(f'Accuracy of the network: {100 * correct // total} %')
                

                classes = ('plastic', 'cardboard', 'metal', 'glass')
                # prepare to count predictions for each class
                correct_pred = {classname: 0 for classname in classes}
                total_pred = {classname: 0 for classname in classes}

                # again no gradients needed
                with torch.no_grad():
                    for data in self.validation_loader:
                        imgs, weight, labels = data['image'].to(self.device), data['weight'].to(self.device), data['class'].to(self.device)
                        outputs = self(imgs, weight)
                        _, predictions = torch.max(outputs, 1)
                        # collect the correct predictions for each class
                        for label, prediction in zip(labels, predictions):
                            if label == prediction:
                                correct_pred[classes[label]] += 1
                            total_pred[classes[label]] += 1

                # print accuracy for each class
                for classname, correct_count in correct_pred.items():
                    accuracy = 100 * float(correct_count) / total_pred[classname]
                    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

            else:
                print("No dataset loaded")

        def infer(self, img, weight):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not torch.is_tensor(img):
                img = self.transform(img)
            if not torch.is_tensor(weight):
                weight = torch.tensor(weight)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            weight = weight.unsqueeze(0)
            weight = weight.to(self.device)
            output = self(img, weight)
            _, predicted = torch.max(output.data, 1)
            return predicted.cpu().data.numpy()[0]
            #return output.detach().cpu().numpy()