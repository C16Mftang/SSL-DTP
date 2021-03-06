import torch
import numpy as np
from tqdm import tqdm
import time
import os
import json

import builder
from conv_data import get_data
from utils import plug_in

# use GPU when possible
use_gpu = True
device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'
print(device)

# acquire the dataset
train, val, test = get_data(data_set='cifar100')

batch_size = 2000
step_size1 = .001
step_size2 = .0002
loss_param = {
    "tau": 0.5,
    "margin": 4,
    "margin_pos": -2,
    "margin_neg": -2,
    "lambda": 4e-3,
    "scale": 1/32
}
sigma = 1.0
lr_targ = 0.5
num_train = 46000
num_epochs = 10
perturb = 0.5
trans = ["aff"]
s_factor = 4
h_factor = 0.2
resume = ''
SS_loss = 'SimCLR'
learning_rule = 'DTP'

def main():
    model = builder.NetDTP(batch_size, step_size1, step_size2, device, loss_param, sigma, lr_targ, loss=SS_loss)
    model.to(device)
    print(model)
    model.train()

    if resume:
        if os.path.isfile(resume):
            print(f"=> loading checkpoint '{resume}'")
            checkpoint = torch.load(resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model.inv_optimizers.load_state_dict(checkpoint['inv_optimizers'])
            for i, fwd_optimizer in enumerate(model.fwd_optimizers):
                fwd_optimizer.load_state_dict(checkpoint['fwd_optimizers'][i])
            print(f"=> loaded checkpoint '{resume}' (epoch {checkpoint['epoch']})")
            # continue training in the same path
            timestr = resume.split('/')[2].strip()
            model_path = os.path.join("../models", timestr)
        else:
            print(f"=> no checkpoint found at '{resume}'")
    else:
        start_epoch = 0
        # if not resume, set up a new path to model and meta data
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join("../models", timestr)
        os.mkdir(model_path)
        
    X = torch.from_numpy(train[0]).to(device)
    X = X.type(torch.FloatTensor)
    n = X.shape[0]
    if (num_train is not None):
        n = np.minimum(n, num_train)

    SS_losses = []
    SS_layer_inv_losses = []
    SS_layer_train_losses = []
    for epoch in range(start_epoch, num_epochs):
        train_loss = 0
        sum_inv_loss = 0
        batch_inv_loss = []
        batch_train_loss = []
        with tqdm(total=2*n) as progress_bar:
            for j in np.arange(0, n, batch_size):
                data = X[j:j+batch_size].to(device)
                # the augmented batch of images
                data_aug = plug_in(data, perturb, trans, s_factor, h_factor, batch_size, device)
                inv_losses, train_losses = model.run_grad(data_aug)

                # get layer wise losses
                inv_loss_all = [l.item() for l in inv_losses]
                batch_inv_loss.append(inv_loss_all)
                train_loss_all = [l.item() for l in train_losses]
                batch_train_loss.append(train_loss_all)
                global_loss = train_losses[-1].item()
                # get overall losses
                sum_inv_loss += np.sum(inv_loss_all)
                train_loss += global_loss

                progress_bar.set_postfix(loss=global_loss)
                progress_bar.update(data_aug.size(0))
        # epoch-wise average loss
        train_loss /= (n//batch_size)
        sum_inv_loss /= (n//batch_size)
        # average layer wise losses to a vector
        layer_inv_loss = np.sum(np.array(batch_inv_loss), axis=0) / (n//batch_size)
        layer_train_loss = np.sum(np.array(batch_train_loss), axis=0) / (n//batch_size)
        print('\nTraining set epoch {}: Avg. train loss: {:.5f}, Avg. inverse loss: {:.5f}'.format(epoch+1, train_loss, sum_inv_loss))
        # save model
        if (epoch+1) % 10 == 0:
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'inv_optimizers' : model.inv_optimizers.state_dict(),
                        'fwd_optimizers': [self.fwd_optimizer1.state_dict(), self.fwd_optimizer2.state_dict(), 
                                           self.fwd_optimizer3.state_dict(), self.fwd_optimizer4.state_dict(), 
                                           self.fwd_optimizer5.state_dict()],
                       }, os.path.join(model_path, f"epoch_{epoch+1}.pth"))
        SS_losses.append(train_loss)
        SS_layer_inv_losses.append(layer_inv_loss)
        SS_layer_train_losses.append(layer_train_loss)

    # save metadata
    np.savez_compressed(os.path.join(model_path, "losses"), train_losses=np.array(SS_losses),
                                                            layer_inv_losses=np.array(SS_layer_inv_losses),
                                                            layer_train_losses=np.array(SS_layer_train_losses))
    var_dict = {}
    for variable in ["batch_size", "step_size", "loss_param", "num_train", "num_epochs", "SS_loss", "learning_rule"]:
        var_dict[variable] = eval(variable)
    with open(os.path.join(model_path, "hparams.json"), 'w') as f:
        json.dump(var_dict, f)

if __name__ == '__main__':
    main()

