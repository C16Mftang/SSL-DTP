import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

def rgb_to_hsv(input, device):
    """Credit to Prof. Yali Amit"""

    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)

    mx, inmx = torch.max(input, dim=1)
    mn, inmc = torch.min(input, dim=1)
    df = mx - mn
    h = torch.zeros(input.shape[0], 1).to(device)
    ii = [0, 1, 2]
    iid = [[1, 2], [2, 0], [0, 1]]
    shift = [360, 120, 240]

    for i, id, s in zip(ii, iid, shift):
        logi = (df != 0) & (inmx == i)
        h[logi, 0] = \
            torch.remainder((60 * (input[logi, id[0]] - input[logi, id[1]]) / df[logi] + s), 360)

    s = torch.zeros(input.shape[0], 1).to(device)
    s[mx != 0, 0] = (df[mx != 0] / mx[mx != 0]) * 100

    v = mx.reshape(input.shape[0], 1) * 100

    output = torch.cat((h / 360., s / 100., v / 100.), dim=1)

    output = output.reshape(sh).transpose(1, 3)
    return output

def hsv_to_rgb(input, device):
    """Credit to Prof. Yali Amit"""

    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)

    hh = input[:, 0]
    hh = hh * 6
    ihh = torch.floor(hh).type(torch.int32)
    ff = (hh - ihh)[:, None];
    v = input[:, 2][:, None]
    s = input[:, 1][:, None]
    p = v * (1.0 - s)
    q = v * (1.0 - (s * ff))
    t = v * (1.0 - (s * (1.0 - ff)));

    output = torch.zeros_like(input).to(device)
    output[ihh == 0, :] = torch.cat((v[ihh == 0], t[ihh == 0], p[ihh == 0]), dim=1)
    output[ihh == 1, :] = torch.cat((q[ihh == 1], v[ihh == 1], p[ihh == 1]), dim=1)
    output[ihh == 2, :] = torch.cat((p[ihh == 2], v[ihh == 2], t[ihh == 2]), dim=1)
    output[ihh == 3, :] = torch.cat((p[ihh == 3], q[ihh == 3], v[ihh == 3]), dim=1)
    output[ihh == 4, :] = torch.cat((t[ihh == 4], p[ihh == 4], v[ihh == 4]), dim=1)
    output[ihh == 5, :] = torch.cat((v[ihh == 5], p[ihh == 5], q[ihh == 5]), dim=1)

    output = output.reshape(sh)
    output = output.transpose(1, 3)
    return output

def deform_data(x_in, perturb, trans, s_factor, h_factor, bsz, device):
    h = x_in.shape[2]
    w = x_in.shape[3]
    nn = x_in.shape[0]
    # [-0.5*perturb, 0.5*perturb]
    u = ((torch.rand(nn,6)-.5)*perturb).to(device)
    # Ammplify the shift part of the deform
    # right now, don't amplify
    u[:,[2,5]] *= 2
    # Just shift and sclae
    #u[:,0]=u[:,4]
    #u[:,[1,3]]=0
    rr = torch.zeros(nn, 6).to(device)
    rr[:, [0,4]] = 1
    # 0,4: scaling; 1,3: rotation; 2,5: shift
    if trans=='shift':
      u[:,[0,1,3,4]] = 0
    elif trans=='scale':
      u[:,[1,3]] = 0
       #+ self.id
    elif 'rotate' in trans:
      u[:,[0,1,3,4]] *= 1.5
      ang = u[:,0]
      v = torch.zeros(nn,6).to(device)
      v[:,0] = torch.cos(ang)
      v[:,1] = -torch.sin(ang)
      v[:,4] = torch.cos(ang)
      v[:,3] = torch.sin(ang)
      s = torch.ones(nn).to(device)
      if 'scale' in trans:
        s = torch.exp(u[:, 1])
      u[:,[0,1,3,4]]=v[:,[0,1,3,4]]*s.reshape(-1,1).expand(nn, 4)
      rr[:, [0,4]]=0
    theta = (u+rr).view(-1, 2, 3)
    grid = F.affine_grid(theta, [nn,1,h,w], align_corners=True)
    x_out = F.grid_sample(x_in, grid, padding_mode='border', align_corners=True)

    if x_in.shape[1]==3 and s_factor>0:
        # change saturation
        v = torch.rand(nn,2).to(device)
        # [2**(-0.5*sfactor), 2**(0.5*sfactor)]
        vv = torch.pow(2, (v[:,0]-.5)*s_factor).reshape(nn, 1, 1)
        # change hue
        uu = ((v[:,1]-.5)*h_factor).reshape(nn, 1, 1)
        x_out_hsv = rgb_to_hsv(x_out, device) # N*3*32*32
        # clamp the 2nd dim of hsv between [0,1]; exceeding=1, below=0
        x_out_hsv[:,1,:,:] = torch.clamp(x_out_hsv[:,1,:,:]*vv, 0., 1.)
        # any value in the 1st dim of hsv thats greater than 1 will be clamped as its remaineder when divided by 1
        x_out_hsv[:,0,:,:] = torch.remainder(x_out_hsv[:,0,:,:]+uu, 1.)
        x_out = hsv_to_rgb(x_out_hsv, device) # N*3*32*32

    # matrix of T/F, with prob=0.5 i.e. half will be flipped
    ii = torch.where(torch.bernoulli(torch.ones(bsz)*.5)==1)
    for i in ii:
        x_out[i] = x_out[i].flip(3)
    return x_out

def plug_in(x_in, perturb, trans, s_factor, h_factor, bsz, device):
    n = x_in.shape[0]
    c = x_in.shape[1]
    h = x_in.shape[2]
    w = x_in.shape[3]
    # asymmetric perturbation
    x_out = deform_data(x_in, perturb, trans, s_factor, h_factor, bsz, device)
    x_out2 = deform_data(x_in, perturb, trans, s_factor, h_factor, bsz, device)
    # create a matrix of length 2n
    x_all = torch.zeros(size=(2*n, c, h, w)).to(device)
    for i in range(n):
        x_all[2*i] = x_out2[i]
        x_all[2*i+1] = x_out[i]
    return x_all

def off_diagonal(x):
    """return a flattened view of the off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()