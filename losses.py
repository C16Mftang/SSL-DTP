import torch
import torch.nn.functional as F
import torch.nn as nn

class SimCLRLoss(nn.Module):

    def __init__(self, tau, batch_size, device):
        super(SimCLRLoss, self).__init__()
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.mask = self.create_mask(batch_size)

    def create_mask(self, batch_size):
        """Positive pair selection"""
        mask = torch.zeros(batch_size*2, batch_size*2).to(self.device)
        for i in range(batch_size):
            mask[2*i, 2*i+1] = 1
            mask[2*i+1, 2*i] = 1
        return mask

    def forward(self, X):
        """
        Input:
            X: embedding created by the base encoder
        Output:
            loss value
        """
        B = X.shape[0]
        d = X.shape[1]
        # normalize each row of X
        X_n = F.normalize(X, dim=1)
        sim_mat = torch.matmul(X_n, torch.transpose(X_n, 0, 1))
        sim_mat = torch.exp(torch.div(sim_mat, self.tau))

        # all diagonal entries should be the same
        row_sums = torch.unsqueeze(torch.sum(sim_mat, dim=1), 1) - sim_mat[0,0]
        loss = -torch.log(torch.div(sim_mat, row_sums))
        loss = torch.sum(loss*self.mask)
        loss = torch.div(loss, B)
        return loss


class NaiveLoss(nn.Module):
    """
    A naive self-supervised loss that only computes 
    the distances between positive pairs
    """
    def __init__(self, batch_size, device):
        super(NaiveLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device

    def forward(self, X):
        B = X.shape[0]
        X1 = X[::2]
        X2 = X[1::2]
        diff = X1 - X2
        loss = torch.sum(torch.relu(diff) + torch.relu(-diff), dim=1)
        loss = torch.mean(torch.div(loss, B))
        return loss
