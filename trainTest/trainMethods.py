
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

def load_weight(model, path, device):
    sd = torch.load(path,map_location=device)
    model.load_state_dict(sd)

def load_weight_stan_boxcars(model, path, device):
    pretrained_dict = torch.load(path,map_location=device)
    pretrained_dict_ids = [0,2,5,7,10,12,14,17,19,21,24,26,28]
    #Add features
    for i in pretrained_dict_ids:
        key='base.features.'+str(i)
        model.state_dict()[key+'.weight'].data.copy_(pretrained_dict[key+'.weight'])
        model.state_dict()[key+'.bias'].data.copy_(pretrained_dict[key+'.bias'])

    # #Add classififers
    # pretrained_dict_ids = [0,3,5.1,6.1]

    # for i in pretrained_dict_ids:
    #     model.state_dict()[key+'.weight'].data.copy_(pretrained_dict[key+'.weight'])
    #     model.state_dict()[key+'.bias'].data.copy_(pretrained_dict[key+'.weight'])

def train_v1(ep, model, optimizer, train_loader, device, config):

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
        loss = F.cross_entropy(pred, target)
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

def train_v2(ep, model, optimizer, train_loader, device, config):
    model.train()

    loss_meter = 0
    acc_meter = 0

    make_loss_meter = 0
    make_acc_meter = 0

    model_loss_meter = 0
    model_acc_meter = 0

    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target,make_target,model_target,submodel_target,generation_target in train_loader:
        data = data.to(device)
        target = target.to(device)
        make_target = make_target.to(device)
        model_target = model_target.to(device)

        optimizer.zero_grad()

        pred, make_pred, model_pred = model(data)
        
        main_loss = F.cross_entropy(pred, target)
        make_loss = F.cross_entropy(make_pred, make_target)
        model_loss = F.cross_entropy(model_pred, model_target)

        loss = main_loss + config['make_loss'] * make_loss + config['model_loss'] * model_loss
        loss.backward()

        optimizer.step()

        #Save accuracy/loss for each feature
        acc = pred.max(1)[1].eq(target).float().mean()
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        model_acc = model_pred.max(1)[1].eq(model_target).float().mean()

        
        loss_meter += loss.item()
        acc_meter += acc.item()

        make_loss_meter += make_loss.item()
        make_acc_meter += make_acc.item()

        model_loss_meter += model_loss.item()
        model_acc_meter += model_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} '

              f'Make L: {make_loss_meter / i:.4f} '
              f'Make A: {make_acc_meter / i:.4f} '

              f'Model L: {model_loss_meter / i:.4f} '
              f'Model A: {model_acc_meter / i:.4f} '
              f'({elapsed:.2f}s)', end='\r')

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)

    make_loss_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)

    model_loss_meter /= len(train_loader)
    model_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        
        'train_make_loss': make_loss_meter,
        'train_make_acc': make_acc_meter,

        'train_model_loss': model_loss_meter,
        'train_model_acc': model_acc_meter,

        'train_time': elapsed
    }

    return trainres

def train_v3(ep, model, optimizer, train_loader, device, config):
    model.train()

    loss_meter = 0
    acc_meter = 0

    make_loss_meter = 0
    make_acc_meter = 0

    model_loss_meter = 0
    model_acc_meter = 0

    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target, make_target, model_target in train_loader:
        data = data.to(device)
        target = target.to(device)
        make_target = make_target.to(device)
        model_target = model_target.to(device)

        optimizer.zero_grad()

        pred, make_pred, model_pred = model(data)
        
        main_loss = F.cross_entropy(pred, target)
        make_loss = F.cross_entropy(make_pred, make_target)
        model_loss = F.cross_entropy(model_pred, model_target)

        loss = main_loss + config['make_loss'] * make_loss + config['make_loss'] * model_loss
        loss.backward()

        optimizer.step()
        
        acc = pred.max(1)[1].eq(target).float().mean()
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        model_acc = model_pred.max(1)[1].eq(model_target).float().mean()


        #Save accuracy/loss for each feature
        loss_meter += loss.item()
        acc_meter += acc.item()

        make_acc_meter += make_acc.item()
        make_loss_meter += make_loss.item()

        model_loss_meter += model_loss.item()
        model_acc_meter += model_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} '

              f'Make L: {make_loss_meter / i:.4f} '
              f'Make A: {make_acc_meter / i:.4f} '
              
              f'Model L: {model_loss_meter / i:.4f} '
              f'Model A: {model_acc_meter / i:.4f} '

              f'({elapsed:.2f}s)', end='\r')

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)

    make_loss_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)

    model_loss_meter /= len(train_loader)
    model_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,

        'train_make_loss': make_loss_meter,
        'train_make_acc': make_acc_meter,

        'train_model_loss': model_loss_meter,
        'train_model_acc': model_acc_meter,

        'train_time': elapsed
    }

    return trainres

def train_v4(ep, model, optimizer, train_loader, device, config):
    model.train()

    loss_meter = 0
    acc_meter = 0

    make_loss_meter = 0
    make_acc_meter = 0

    model_loss_meter = 0
    model_acc_meter = 0

    submodel_loss_meter = 0
    submodel_acc_meter = 0
    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target,make_target,model_target,submodel_target,generation_target in train_loader:
        data = data.to(device)
        target = target.to(device)
        make_target = make_target.to(device)
        model_target = model_target.to(device)
        submodel_target = submodel_target.to(device)

        optimizer.zero_grad()

        pred, make_pred, model_pred,submodel_pred = model(data)
        
        main_loss = F.cross_entropy(pred, target)
        make_loss = F.cross_entropy(make_pred, make_target)
        model_loss = F.cross_entropy(model_pred, model_target)
        submodel_loss = F.cross_entropy(submodel_pred, submodel_target)

        loss = main_loss + config['make_loss'] * make_loss + config['model_loss'] * model_loss + config['submodel_loss'] * submodel_loss
        loss.backward()

        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        model_acc = model_pred.max(1)[1].eq(model_target).float().mean()
        submodel_acc = submodel_pred.max(1)[1].eq(submodel_target).float().mean()

        #Main
        loss_meter += loss.item()
        acc_meter += acc.item()

        #Make
        make_loss_meter += make_loss.item()
        make_acc_meter += make_acc.item()

        #Model
        model_loss_meter += model_loss.item()
        model_acc_meter += model_acc.item()

        #Submodel
        submodel_loss_meter += submodel_loss.item()
        submodel_acc_meter += submodel_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} '
            
              f'Make L: {make_loss_meter / i:.4f} '
              f'Make A: {make_acc_meter / i:.4f} '

              f'Model L: {model_loss_meter / i:.4f} '
              f'Model A: {model_acc_meter / i:.4f} '
              
              f'SubModel L: {submodel_loss_meter / i:.4f} '
              f'SubModel A: {submodel_acc_meter / i:.4f} '
              f'({elapsed:.2f}s)', end='\r')

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)

    make_loss_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)

    model_loss_meter /= len(train_loader)
    model_acc_meter /= len(train_loader)

    submodel_loss_meter /= len(train_loader)
    submodel_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,

        'train_make_loss': make_loss_meter,
        'train_make_acc':make_acc_meter,

        'train_model_loss': model_loss_meter,
        'train_model_acc':model_acc_meter,

        'submodel_acc_loss':submodel_loss_meter,
        'submodel_acc_acc':submodel_acc_meter,

        'train_time': elapsed
    }

    return trainres

# ---------------------------
# Classic multitask learning
# ---------------------------

#Predicit each feature for label and backpropogate with combined loss
def train_v5(ep, model, optimizer, train_loader, device, config):
    model.train()

    loss_meter = 0
    acc_meter = 0

    make_loss_meter = 0
    make_acc_meter = 0

    model_loss_meter = 0
    model_acc_meter = 0

    submodel_loss_meter = 0
    submodel_acc_meter = 0

    generation_loss_meter = 0
    generation_acc_meter = 0

    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target,make_target,model_target,submodel_target,generation_target in train_loader:
        data = data.to(device)
        target = target.to(device)
        make_target = make_target.to(device)
        model_target = model_target.to(device)
        submodel_target = submodel_target.to(device)
        generation_target = generation_target.to(device)

        optimizer.zero_grad()

        make_pred, model_pred,submodel_pred,generation_pred = model(data)
        
        #Calucalate individual loss for part labels
        make_loss = F.cross_entropy(make_pred, make_target)
        model_loss = F.cross_entropy(model_pred, model_target)
        submodel_loss = F.cross_entropy(submodel_pred, submodel_target)
        generation_loss = F.cross_entropy(generation_pred, generation_target)
        
        loss = config['make_loss'] * make_loss + config['model_loss'] * model_loss + config['submodel_loss'] * submodel_loss + config['generation_loss'] * generation_loss
        loss.backward()

        optimizer.step()
        
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        model_acc = model_pred.max(1)[1].eq(model_target).float().mean()
        submodel_acc = submodel_pred.max(1)[1].eq(submodel_target).float().mean()
        generation_acc = generation_pred.max(1)[1].eq(generation_target).float().mean()

        #Main
        loss_meter += loss.item()
        # acc_meter += acc.item() No used

        #Make
        make_loss_meter += make_loss.item()
        make_acc_meter += make_acc.item()

        #Model
        model_loss_meter += model_loss.item()
        model_acc_meter += model_acc.item()

        #Submodel
        submodel_loss_meter += submodel_loss.item()
        submodel_acc_meter += submodel_acc.item()

        #Generation
        generation_loss_meter += generation_acc.item()
        generation_acc_meter += generation_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '

              f'Loss: {loss_meter / i:.4f} '
            #   f'Acc: {acc_meter / i:.4f} '

              f'Make L: {make_loss_meter / i:.4f} '
              f'Make A: {make_acc_meter / i:.4f} '

              f'Model L: {model_loss_meter / i:.4f} '
              f'Model A: {model_acc_meter / i:.4f} '
              
              f'SubModel L: {submodel_loss_meter / i:.4f} '
              f'SubModel A: {submodel_acc_meter / i:.4f} '

              f'Generation L: {generation_loss_meter / i:.4f} '
              f'Generation A: {generation_acc_meter / i:.4f} '

              f'({elapsed:.2f}s)', end='\r')

    print()

    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)

    make_loss_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)

    model_loss_meter /= len(train_loader)
    model_acc_meter /= len(train_loader)

    submodel_loss_meter /=len(train_loader)
    submodel_acc_meter /=len(train_loader)

    generation_loss_meter /= len(train_loader)
    generation_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,#Not used so will be set to 0. Kept so that the headers don't change in excel sheet. Easier to graph

        'train_make_loss': make_loss_meter,
        'train_make_acc':make_acc_meter,

        'train_model_loss': model_loss_meter,
        'train_model_acc':model_acc_meter,

        'submodel_acc_loss':submodel_loss_meter,
        'submodel_acc_acc':submodel_acc_meter,

        'generation_acc_loss':generation_loss_meter,
        'generation_acc_acc':generation_acc_meter,

        'train_time': elapsed
    }

    return trainres

#ML Learning but train best model for just make
def train_v6(ep, model, optimizer, train_loader, device, config):
    model.train()

    loss_meter = 0
    make_loss_meter = 0
    make_acc_meter = 0

    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target,make_target,model_target,submodel_target,generation_target in train_loader:
        data = data.to(device)
        make_target = make_target.to(device)

        optimizer.zero_grad()

        make_pred, model_pred,submodel_pred,generation_pred = model(data)
        
        #Calucalate individual loss for part labels
        make_loss = F.cross_entropy(make_pred, make_target)
        
        
        loss = config['make_loss'] * make_loss
        loss.backward()

        optimizer.step()
        
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()

        loss_meter += loss.item()
        make_loss_meter += make_loss.item()
        make_acc_meter += make_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Make L: {make_loss_meter / i:.4f} '
              f'Make A: {make_acc_meter / i:.4f} '

              f'({elapsed:.2f}s)', end='\r')

    print()
    loss_meter /= len(train_loader)
    make_loss_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)


    trainres = {
        'train_loss': loss_meter,
        'train_acc': -1,#Not used so will be set to 0. Kept so that the headers don't change in excel sheet. Easier to graph

        'train_make_loss': make_loss_meter,
        'train_make_acc':make_acc_meter,

        'train_model_loss': -1,
        'train_model_acc':-1,

        'submodel_acc_loss':-1,
        'submodel_acc_acc':-1,

        'generation_acc_loss':-1,
        'generation_acc_acc':-1,

        'train_time': elapsed
    }
    return trainres

#ML Learning but train best model for just model
def train_v7(ep, model, optimizer, train_loader, device, config):
    model.train()

    loss_meter = 0
    model_loss_meter = 0
    model_acc_meter = 0

    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target,make_target,model_target,submodel_target,generation_target in train_loader:
        data = data.to(device)
        model_target = model_target.to(device)

        optimizer.zero_grad()

        make_pred, model_pred,submodel_pred,generation_pred = model(data)
        
        #Calucalate individual loss for part labels
        model_loss = F.cross_entropy(model_pred, model_target)
        
        loss = config['model_loss'] * model_loss
        loss.backward()

        optimizer.step()
        
        model_acc = model_pred.max(1)[1].eq(model_target).float().mean()

        loss_meter += loss.item()
        model_loss_meter += model_loss.item()
        model_acc_meter += model_acc.item()
        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Model L: {model_loss_meter / i:.4f} '
              f'Model A: {model_acc_meter / i:.4f} '
              f'({elapsed:.2f}s)', end='\r')

    print()

    loss_meter /= len(train_loader)
    model_loss_meter /= len(train_loader)
    model_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': -1,#Not used so will be set to 0. Kept so that the headers don't change in excel sheet. Easier to graph

        'train_make_loss': -1,
        'train_make_acc':-1,

        'train_model_loss': model_loss_meter,
        'train_model_acc':model_acc_meter,

        'submodel_acc_loss':-1,
        'submodel_acc_acc':-1,

        'generation_acc_loss':-1,
        'generation_acc_acc':-1,

        'train_time': elapsed
    }

    return trainres

#ML Learning but train best model for just submodel
def train_v8(ep, model, optimizer, train_loader, device, config):
    model.train()

    loss_meter = 0
    submodel_loss_meter = 0
    submodel_acc_meter = 0
    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target,make_target,model_target,submodel_target,generation_target in train_loader:
        data = data.to(device)
        submodel_target = submodel_target.to(device)

        optimizer.zero_grad()

        make_pred, model_pred,submodel_pred,generation_pred = model(data)
        
        #Calucalate individual loss for part labels
        submodel_loss = F.cross_entropy(submodel_pred, submodel_target)
        
        loss = config['submodel_loss'] * submodel_loss
        loss.backward()

        optimizer.step()
        
        submodel_acc = submodel_pred.max(1)[1].eq(submodel_target).float().mean()

        loss_meter += loss.item()
        submodel_loss_meter += submodel_loss.item()
        submodel_acc_meter += submodel_acc.item()
        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'SubModel L: {submodel_loss_meter / i:.4f} '
              f'SubModel A: {submodel_acc_meter / i:.4f} '
              f'({elapsed:.2f}s)', end='\r')

    print()

    loss_meter /= len(train_loader)
    submodel_loss_meter /=len(train_loader)
    submodel_acc_meter /=len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': -1,#Not used so will be set to 0. Kept so that the headers don't change in excel sheet. Easier to graph

        'train_make_loss': -1,
        'train_make_acc':-1,

        'train_model_loss': -1,
        'train_model_acc':-1,

        'submodel_acc_loss':submodel_loss_meter,
        'submodel_acc_acc':submodel_acc_meter,

        'generation_acc_loss':-1,
        'generation_acc_acc':-1,

        'train_time': elapsed
    }

    return trainres

#ML Learning but train best model for just generation
def train_v9(ep, model, optimizer, train_loader, device, config):
    model.train()

    loss_meter = 0
    generation_loss_meter = 0
    generation_acc_meter = 0

    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target,make_target,model_target,submodel_target,generation_target in train_loader:
        data = data.to(device)
        generation_target = generation_target.to(device)

        optimizer.zero_grad()

        make_pred, model_pred,submodel_pred,generation_pred = model(data)
        
        #Calucalate individual loss for part labels
        generation_loss = F.cross_entropy(generation_pred, generation_target)
        
        loss = config['generation_loss'] * generation_loss
        loss.backward()

        optimizer.step()
        
        generation_acc = generation_pred.max(1)[1].eq(generation_target).float().mean()

        loss_meter += loss.item()
        generation_loss_meter += generation_acc.item()
        generation_acc_meter += generation_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Generation L: {generation_loss_meter / i:.4f} '
              f'Generation A: {generation_acc_meter / i:.4f} '

              f'({elapsed:.2f}s)', end='\r')

    print()

    loss_meter /= len(train_loader)
    generation_loss_meter /= len(train_loader)
    generation_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': -1,

        'train_make_loss': -1,
        'train_make_acc':-1,

        'train_model_loss': -1,
        'train_model_acc':-1,

        'submodel_acc_loss':-1,
        'submodel_acc_acc':-1,

        'generation_acc_loss':generation_loss_meter,
        'generation_acc_acc':generation_acc_meter,

        'train_time': elapsed
    }

    return trainres

# -------------------------------
# Classic multitask learning END
# -------------------------------