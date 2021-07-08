import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os

import builder
from conv_data import get_data

# use GPU when possible
use_gpu = True
device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'
print(device)

# acquire the dataset
train, val, test = get_data(data_set='cifar10')

test_step_size = 0.001
num_epochs = 100
num_train = 46000
batch_size = 1000
pretrained = '../models/20210706-132711/epoch_400.pth'

def main():
    lcf_model = builder.NewNet(test_step_size, p=.0)
    lcf_model.to(device)
    print(lcf_model)
    for name, param in lcf_model.named_parameters():
        if name not in ['fc10.weight', 'fc10.bias']:
            param.requires_grad = False

    X = torch.from_numpy(train[0]).to(device)
    X = X.type(torch.FloatTensor)
    y = torch.from_numpy(train[1]).to(device)
    y = y.type(torch.long)
    n = X.shape[0]
    if (num_train is not None):
        n = np.minimum(n, num_train)
    
    if os.path.isfile(pretrained):
        print(f"=> loading checkpoint '{pretrained}'")
        checkpoint = torch.load(pretrained, map_location=device)
        state_dict = checkpoint['state_dict']
        msg = lcf_model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc10.weight", "fc10.bias"}
        print(f"=> loaded pre-trained model '{pretrained}'")

        for epoch in range(num_epochs):
            # training
            lcf_model.train()
            train_loss=0
            train_correct=0
            with tqdm(total=n) as progress_bar:
                for j in np.arange(0, n, batch_size):
                    data = X[j:j+batch_size].to(device)
                    targ = y[j:j+batch_size].to(device)
                    loss, correct = lcf_model.run_grad(data, targ) 
                    train_loss += loss.item()
                    train_correct += correct.item()
                    progress_bar.set_postfix(loss=loss.item())
                    progress_bar.update(data.size(0))
                train_loss /= (len(y) // batch_size)
                print('\nTraining set epoch {}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                        epoch+1,
                        train_loss, 
                        train_correct, 
                        n,
                        100. * train_correct / n
                     ))
            # validation set
            validation(lcf_model, val)
        # test set
        validation(lcf_model, test, ttype='Test')
            
    else:
        print(f"=> no checkpoint found at '{pretrained}'")

def validation(model, val_data, ttype='Validation'):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_correct = 0
        vald = val_data[0]
        yval = val_data[1]
        for j in np.arange(0, len(yval), batch_size):
            data = torch.from_numpy(vald[j:j+batch_size]).type(torch.cuda.FloatTensor).to(device)
            targ = torch.from_numpy(yval[j:j+batch_size]).type(torch.long).to(device)
            loss, correct = model.get_acc_and_loss(data, targ)
            test_loss += loss.item()
            test_correct += correct.item()

        test_loss /= (len(yval) // batch_size)
        print('\n{} set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                ttype,
                test_loss, 
                test_correct, 
                len(yval),
                100. * test_correct / len(yval)
             ))

if __name__ == '__main__':
    main()
