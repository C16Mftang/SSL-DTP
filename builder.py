import torch
import torchvision
from torchvision import datasets
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

import losses

class NewNet(nn.Module):
    """A supervised network with only the last layer trainable for linear evaluation"""
    def __init__(self, step_size=0.003, p=.0):
        super(NewNet, self).__init__()
        # forward functions
        self.f1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.f2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.f3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.f4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # use 10 here
        self.fc10 = nn.Linear(1024, 10)
        self.conv2_drop = nn.Dropout2d(p)
        self.optimizer = torch.optim.Adam(list(self.fc10.parameters()), lr=step_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        h1 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f1(x)), 2))
        h2 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f2(h1)), 2))
        h3 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f3(h2)), 2))
        h4 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f4(h3)), 2))
        output = self.fc10(h4.view(-1, 1024))
        return output

    def get_acc_and_loss(self, x, targ):
        output = self.forward(x)
        loss = self.criterion(output, targ)
        pred = torch.max(output, 1)[1]
        correct = torch.eq(pred, targ).sum()
        return loss, correct

    def run_grad(self, x, targ):
        loss, correct = self.get_acc_and_loss(x, targ)
        self.optimizer.zero_grad()
        d_f = grad(outputs=loss, inputs=self.fc10.parameters(), retain_graph=True)
        dw, db = d_f[0].clone(), d_f[1].clone()
        self.fc10.weight.grad = dw
        self.fc10.bias.grad = db
        self.optimizer.step()
        return loss, correct

class NetBP(nn.Module):
    """The base encoder net trained with BP, for CIFAR"""
    def __init__(self, batch_size, step_size, device, loss_param, p=0., loss='SimCLR'):
        super(NetBP, self).__init__()
        self.batch_size = batch_size
        self.loss = loss
        self.f1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.f2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.f3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.f4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.f5 = nn.Linear(1024, 64)
        self.conv2_drop = nn.Dropout2d(p)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=step_size)
        if loss == 'Naive':
            self.criterion = losses.NaiveLoss(batch_size=batch_size, device=device)
        elif loss == 'SimCLR':
            self.criterion = losses.SimCLRLoss(tau=loss_param['tau'], batch_size=batch_size, device=device)

    def forward(self, x):
        """
        Input:
            x: a batch of image
        """
        h1 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f1(x)), 2))
        h2 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f2(h1)), 2))
        h3 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f3(h2)), 2))
        h4 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f4(h3)), 2))
        output = torch.tanh(self.f5(h4.view(-1, 1024)))
        return output
    
    def get_loss(self, data):
        output = self.forward(data)
        loss = self.criterion(output)
        return loss
        
    def run_grad(self, data):
        loss = self.get_loss(data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss