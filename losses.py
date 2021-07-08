import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import off_diagonal

class SimCLRLoss(nn.Module):
    """
    The NT-Xent loss used in https://arxiv.org/abs/2002.05709
    Inputs:
        tau: temperature parameter
    """
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

class SSHingeLoss(nn.Module):
    """
    The contrastive hinge loss, which requires no normalization
    Inputs:
        margin_pos: the margin used for selecting hard positives
        margin_neg: the margin used for selection hard negatives
    """
    def __init__(self, margin_pos, margin_neg, batch_size, device):
        super(SSHingeLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.mask = self.create_mask1(batch_size)
        self.mask2 = self.create_mask2(batch_size)
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def create_mask1(self, batch_size):
        mask = torch.zeros(batch_size*2, batch_size*2).to(self.device)
        for i in range(batch_size):
            mask[2*i, 2*i+1] = 1
            mask[2*i+1, 2*i] = 1
        return mask
    
    def create_mask2(self, batch_size):
        mask2 = torch.ones(batch_size*2,batch_size*2).to(self.device)
        torch.diagonal(mask2).fill_(0)
        for i in range(batch_size):
            mask2[2*i, 2*i+1] = 0
            mask2[2*i+1, 2*i] = 0
        return mask2

    def forward(self, X):
        B = X.shape[0]
        X_n = X
        l1_dist_vec = torch.pdist(X_n, p=1)
        l1_dist_mat = torch.zeros((B, B)).to(self.device)
        triu_indices = torch.triu_indices(row=B, col=B, offset=1)
        l1_dist_mat[triu_indices[0], triu_indices[1]] = l1_dist_vec
        d = l1_dist_mat + l1_dist_mat.t()

        # select the positive distances
        mask_boo = self.mask > 0
        pos_dist = torch.masked_select(d, mask_boo).view(-1,1)
        pos_dist = torch.relu(pos_dist + 1 + self.margin_pos)

        # select the negative distances
        mask_boo2 = self.mask2 > 0
        neg_dist = torch.masked_select(d, mask_boo2)
        neg_dist = neg_dist.reshape(d.shape[0], d.shape[1]-2)
        neg_dist = torch.relu(1 - self.margin_neg - neg_dist)
        neg_dist = torch.sum(neg_dist, dim=1, keepdim=True) / (torch.sum(neg_dist>0, dim=1, keepdim=True)+1e-12)

        loss = torch.mean(pos_dist + neg_dist)
        return loss

class BarlowTwinsLoss(nn.Module):
    """
    The Barlow Twins loss used in https://arxiv.org/abs/2103.03230
    Inputs:
        lambd: a weighing hyper-parameter between diagonal and off-diagonal entries
        scale: a rather mysterious hyper-parameter not explained well in the paper
    """
    def __init__(self, lambd, scale, batch_size, device):
        super(BarlowTwinsLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.lambd = lambd
        self.scale = scale
        self.bn = nn.BatchNorm1d(64, affine=False, track_running_stats=True)

    def forward(self, X):
        x1 = X[::2]
        x2 = X[1::2]

        # empirical cross-correlation matrix
        c = self.bn(x1).T @ self.bn(x2)
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale)
        loss = on_diag + self.lambd * off_diag
        return loss

class NaiveLoss(nn.Module):
    """A naive self-supervised loss that only computes the distances between positive pairs"""
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
