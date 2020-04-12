
import argparse
import os
import pprint as pp
import time

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import time

from trainTest.confusionMatrix import update_confusion_matrix

# pred = torch.max(pred,1).indices CHANGE TO SPEED UP TIMES

def dual_cross_entropy(pred, target,alpha=1,beta=4.5):    
    Lce = F.cross_entropy(pred, target)
    target = torch.eye(107)[target]
    logsoftmax = nn.LogSoftmax()
    lr = torch.mean(torch.sum(-(1-target) * logsoftmax(alpha*pred), dim=1))
   
    return Lce + (beta * lr)
    

def train_v7(ep, model, optimizer, train_loader, device, config,loss_function):

    print("---------Training-------")

    model.train() # Set model to training mode

    loss_meter = 0
    acc_meter = 0
    i = 0

    start_time = time.time()
    elapsed = 0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

         # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = model(data)
        loss = dual_cross_entropy(pred, target)
        loss.backward()#How does this instigate back propogation
        optimizer.step()#updates parameters

        acc = pred.max(1)[1].eq(target).float().mean()
        

        loss_meter += loss.item()
        acc_meter += acc.item()
        i += 1
        elapsed = time.time() - start_time
    

        #Moved this out of for as I don't watch it all the time and will speed up performace
        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
        f'Loss: {loss_meter / i:.4f} '
        f'Acc: {acc_meter / i:.4f} ({elapsed:.2f}s)'
        ,end='\r')

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_time': elapsed,
    }

    return trainres

def test_v7(model, test_loader, device, config,confusion_matrix):
    model.eval()

    loss_meter = 0
    acc_meter = 0
    runcount = 0
    elapsed = 0
   
    i = 0
    
    with torch.no_grad():
        start_time = time.time()
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            pred = model(data)

            loss = loss_function(pred, target) * data.size(0)
            acc = pred.max(1)[1].eq(target).float().sum()
            if (not confusion_matrix==None):
                update_confusion_matrix(confusion_matrix['total'],pred,target)
          
            loss_meter += loss.item()
            acc_meter += acc.item()
            i += 1
            elapsed = time.time() - start_time
            runcount += data.size(0)
        #Moved this out of for as I don't watch it all the time and will speed up performace
        print(f'[{i}/{len(test_loader)}]: '
                f'Loss: {loss_meter / runcount:.4f} '
                f'Acc: {acc_meter / runcount:.4f} ({elapsed:.2f}s)'
                , end='\r')

        print()

        loss_meter /= runcount
        acc_meter /= runcount

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        'val_time': elapsed,
    }

    # print(f'Test Result: Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} ({elapsed:.2f}s)')# printed twice

    return valres