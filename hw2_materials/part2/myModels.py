# ref:https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
# https://github.com/pytorch/vision/blob/f5afae50bc8e99b873e2345bcda2dedfc863a737/torchvision/models/resnet.py#L182

# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet
import cv2

import torch
import torch.nn as nn

import torchvision
from torchinfo import summary



class Resnext50(nn.Module):
    def __init__(self, num_out):
        super(Resnext50, self).__init__()
        self.resnext50 = torchvision.models.resnext50_32x4d(pretrained = True)
        self.resnext50.fc = nn.Linear(self.resnext50.fc.in_features, num_out)
    def forward(self, x):
        x = self.resnext50(x)

        return x

class Resnet18(nn.Module):
    def __init__(self, num_out):
        super(Resnet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_out)
    def forward(self, x):
        x = self.resnet18(x)
        return x

class Resnet50(nn.Module):
    def __init__(self, num_out):
        super(Resnet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(in_features=self.resnet50.fc.in_features, out_features = num_out)
        #change resnet50.backbone
    def forward(self, x):
        x = self.resnet50(x)
        return x


class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        return x

    
class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        self.in_channels, self.activation = in_channels, nn.ReLU()

        # self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
        #                            nn.BatchNorm2d(in_channels))
        self.blocks = nn.Identity()
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        # x = self.conv1(x)
        residual = x
        x = self.blocks(x)
        x += residual
        x = self.activation(x)
        return x
    # No need! Because of the same dimension
    # def should_apply_shortcut(self):
    #     return self.in_channels != self.out_channels

        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.layer1 = residual_block(in_channels=64)
        self.layer2 = residual_block(in_channels=32)
        self.layer3 = residual_block(in_channels=16)

        self.conv1 = nn.Sequential(self.stem_conv,
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=120, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(1920, 480), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(480, 240), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(240,120), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc5 = nn.Linear(84,num_out)
        
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
        
        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.layer3(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        
        return x

# resnext50 = Resnext50(10)
# summary(resnext50)