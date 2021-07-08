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
    """
    The base encoder net trained with BP, for CIFAR
    Inputs:
        step_size: learning rate for backprop
        loss_param: hyper-parameters for different SS losses
    """
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
        elif loss == 'Hinge':
            self.criterion = losses.SSHingeLoss(loss_param['margin_pos'], loss_param['margin_neg'], 
                                                batch_size=batch_size, device=device)

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

class NetDTP(nn.Module):
    """
    The base encoder net trained with DTP, for CIFAR
    Input:
        step_size1: learning rate for the layer-wise autoencoders in TP
        step_size2: learning rate for training the feedforward weights
        loss_param: hyper-parameters for different SS losses
        sigma: std of noise added to train the layer-wise auto-encoders
        lr_targ: learning rate used to compute the first (top-layer) target
    
    """
    def __init__(self, batch_size, step_size1, step_size2, device, loss_param, sigma=1, lr_targ=0.5, p=0., loss='SimCLR'):
        super(NetDTP, self).__init__()
        self.sigma = sigma
        self.lr_targ = lr_targ
        self.loss = loss
        # forward functions
        self.f1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.f2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.f3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.f4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.f5 = nn.Linear(1024, 64)

        # inverse functions to be trained
        self.g2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.g3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.g4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv2_drop = nn.Dropout2d(p)
        inv_params = list(self.g2.parameters()) + list(self.g3.parameters()) + list(self.g4.parameters())
        self.inv_optimizers = torch.optim.Adam(inv_params, lr=step_size1)
        self.fwd_optimizer1 = torch.optim.Adam(list(self.f1.parameters()), lr=step_size2)
        self.fwd_optimizer2 = torch.optim.Adam(list(self.f2.parameters()), lr=step_size2)
        self.fwd_optimizer3 = torch.optim.Adam(list(self.f3.parameters()), lr=step_size2)
        self.fwd_optimizer4 = torch.optim.Adam(list(self.f4.parameters()), lr=step_size2)
        self.fwd_optimizer5 = torch.optim.Adam(list(self.f5.parameters()), lr=step_size2)
        self.fwd_optimizers = [self.fwd_optimizer1, self.fwd_optimizer2, self.fwd_optimizer3, self.fwd_optimizer4, self.fwd_optimizer5]

        # contrastive global loss
        if loss == 'Naive':
            self.global_criterion = losses.NaiveLoss(batch_size=batch_size, device=device)
        elif loss == 'SimCLR':
            self.global_criterion = losses.SimCLRLoss(tau=loss_param['tau'], batch_size=batch_size, device=device)
        elif loss == 'Hinge':
            self.criterion = losses.SSHingeLoss(loss_param['margin_pos'], loss_param['margin_neg'], 
                                                batch_size=batch_size, device=device)
        # local loss
        self.local_criterion = nn.MSELoss()

    def forward(self, x):
        h1 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f1(x)), 2))
        h2 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f2(h1)), 2))
        h3 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f3(h2)), 2))
        h4 = torch.tanh(F.max_pool2d(self.conv2_drop(self.f4(h3)), 2))
        output = torch.tanh(self.f5(h4.view(-1, 1024)))
        forwards = [h1, h2, h3, h4, output]
        return forwards

    def global_loss(self, x):
        forwards = self.forward(x)
        output = forwards[-1]
        loss = self.global_criterion(output)
        return forwards, loss

    def get_targets(self, x):
        forwards, loss = self.global_loss(x)
        h1, h2, h3, h4, output = forwards
        
        # get the targets for all layers
        h4.retain_grad()
        d_h4 = grad(outputs=loss, inputs=h4, retain_graph=True)
        h4_ = h4 - self.lr_targ * d_h4[0]
        h3_ = h3 - torch.tanh(self.g4(h4)) + torch.tanh(self.g4(h4_))
        h2_ = h2 - torch.tanh(self.g3(h3)) + torch.tanh(self.g3(h3_))
        h1_ = h1 - torch.tanh(self.g2(h2)) + torch.tanh(self.g2(h2_))

        # do not calculate gradients on the targets
        targets = [h1_.clone().detach(), h2_.clone().detach(), h3_.clone().detach(), h4_.clone().detach()]
        return forwards, loss, targets
    
    def train_inverse(self, forwards):
        """train the approximate inverses g to make it close to f^(-1)"""
        h1, h2, h3, h4, output = forwards

        self.inv_optimizers.zero_grad()

        # corrupted pairs
        h3_c = (h3 + self.sigma*torch.randn(h3.shape).to(device))
        fh3_c = torch.tanh(F.max_pool2d(self.f4(h3_c), 2))
        L3 = self.local_criterion(torch.tanh(self.g4(fh3_c)), h3_c)
        # gradients and update
        d_g4 = grad(outputs=L3, inputs=self.g4.parameters(), retain_graph=True)
        dw4, db4 = d_g4[0].clone(), d_g4[1].clone()
        self.g4.weight.grad = dw4
        self.g4.bias.grad = db4

        h2_c = (h2 + self.sigma*torch.randn(h2.shape).to(device))
        fh2_c = torch.tanh(F.max_pool2d(self.f3(h2_c), 2))
        L2 = self.local_criterion(torch.tanh(self.g3(fh2_c)), h2_c)
        # gradients and update
        d_g3 = grad(outputs=L2, inputs=self.g3.parameters(), retain_graph=True)
        dw3, db3 = d_g3[0].clone(), d_g3[1].clone()
        self.g3.weight.grad = dw3
        self.g3.bias.grad = db3
        
        h1_c = h1 + self.sigma*torch.randn(h1.shape).to(device)
        fh1_c = torch.tanh(F.max_pool2d(self.f2(h1_c), 2))
        L1 = self.local_criterion(torch.tanh(self.g2(fh1_c)), h1_c)
        # gradients and update
        d_g2 = grad(outputs=L1, inputs=self.g2.parameters())
        dw2, db2 = d_g2[0].clone(), d_g2[1].clone()
        self.g2.weight.grad = dw2
        self.g2.bias.grad = db2

        self.inv_optimizers.step()
        return [L1, L2, L3]

    def run_grad(self, x):
        forwards, global_loss, targets = self.get_targets(x)
        # update g_i
        inv_losses = self.train_inverse(forwards)

        h1, h2, h3, h4 = forwards[:-1]
        h1_, h2_, h3_, h4_ = targets

        # targets are deemed as constants here
        L1 = self.local_criterion(h1, h1_)
        L2 = self.local_criterion(h2, h2_)
        L3 = self.local_criterion(h3, h3_)
        L4 = self.local_criterion(h4, h4_)

        self.fwd_optimizer1.zero_grad()
        d_f1 = grad(outputs=L1, inputs=self.f1.parameters(), retain_graph=True)
        dw1, db1 = d_f1[0].clone(), d_f1[1].clone()
        self.f1.weight.grad = dw1
        self.f1.bias.grad = db1
        self.fwd_optimizer1.step()

        self.fwd_optimizer2.zero_grad()
        d_f2 = grad(outputs=L2, inputs=self.f2.parameters(), retain_graph=True)
        dw2, db2 = d_f2[0].clone(), d_f2[1].clone()
        self.f2.weight.grad = dw2
        self.f2.bias.grad = db2 
        self.fwd_optimizer2.step()

        self.fwd_optimizer3.zero_grad()
        d_f3 = grad(outputs=L3, inputs=self.f3.parameters(), retain_graph=True)
        dw3, db3 = d_f3[0].clone(), d_f3[1].clone()
        self.f3.weight.grad = dw3
        self.f3.bias.grad = db3
        self.fwd_optimizer3.step()

        self.fwd_optimizer4.zero_grad()
        d_f4 = grad(outputs=L4, inputs=self.f4.parameters(), retain_graph=True)
        dw4, db4 = d_f4[0].clone(), d_f4[1].clone()
        self.f4.weight.grad = dw4
        self.f4.bias.grad = db4
        self.fwd_optimizer4.step()

        self.fwd_optimizer5.zero_grad()
        d_f5 = grad(outputs=global_loss, inputs=self.f5.parameters())
        dw5, db5 = d_f5[0].clone(), d_f5[1].clone()
        self.f5.weight.grad = dw5
        self.f5.bias.grad = db5
        self.fwd_optimizer5.step()

        training_losses = [L1, L2, L3, L4, global_loss]
        return inv_losses, training_losses