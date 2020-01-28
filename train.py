from datasets import prepare_loader
from models import construct_model
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import pandas as pd
import pprint as pp
# from sklearn.metrics import precision_score,f1_score,recall_score

def train_v1(ep, model, optimizer, lr_scheduler, train_loader, device, config):

    lr_scheduler.step()
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
        loss.backward()
        optimizer.step()#updates parameters

        acc = pred.max(1)[1].eq(target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        i += 1
        elapsed = time.time() - start_time

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

def train_v2(ep, model, optimizer, lr_scheduler, train_loader, device, config):

    lr_scheduler.step()
    print("---------Training-------")

    model.train() # Set model to training mode

    loss_meter = 0
    acc_meter = 0
    i = 0

    start_time = time.time()
    elapsed = 0
    for data, target,make_target,model_target,submodel_target,generation_target in train_loader:
        data = data.to(device)
        target = target.to(device)
        make_target = target.to(device)
        model_target = model_target.to(device)
        submodel_target = submodel_target.to(device)
        # generation_target = generation_target.to(device) #TODO ADD THIS ONE TOO 

         # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred, make_pred, model_pred,submodel_pred = model(data)

        loss_main = F.cross_entropy(pred, target)
        loss_make = F.cross_entropy(make_pred, make_target)
        loss_model = F.cross_entropy(model_pred, model_target)
        loss_submodel = F.cross_entropy(submodel_pred, submodel_target)

        loss = loss_main + config['make_loss'] * loss_make + config['model_loss'] * loss_model * config['submodel_loss'] * loss_model
        loss.backward()
        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        model_acc = model_pred.max(1)[1].eq(model_target).float().mean()
        submodel_acc = submodel_pred.max(1)[1].eq(submodel_target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        make_acc_meter += make_acc.item()
        model_acc_meter += model_acc.item()
        submodel_acc_meter += submodel_acc.item()


        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} ({elapsed:.2f}s)'
              f'Make: {make_acc_meter / i:.4f} '
              f'model: {model_acc_meter / i:.4f} '
              f'Subtype: {submodel_acc_meter / i:.4f} '
              ,end='\r')

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)
    model_acc_meter /= len(train_loader)
    submodel_acc_meter /= len(train_loader)


    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_make_acc': make_acc_meter,
        'train_model_acc': model_acc_meter,
        'train_submodel_acc': submodel_acc_meter,
        'train_time': elapsed,
    }

    return trainres

def test(model, test_loader, device, config):
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

            loss = F.cross_entropy(pred, target) * data.size(0)
            acc = pred.max(1)[1].eq(target).float().sum()

            loss_meter += loss.item()
            acc_meter += acc.item()
            i += 1
            elapsed = time.time() - start_time
            runcount += data.size(0)
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
        # 'val_percision':percision_meter,
        # 'val_recall':recall_meter,
        # 'val_f1':f1_meter,
    }

    print(f'Test Result: Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} ({elapsed:.2f}s)')

    return valres


exp_dir = 'saves'


def main():
    # TODO what does this do
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # Set up config
    config = {
        'batch_size': 32,
        'test_batch_size': 32,
        'lr': 0.001, #TODO change this
        'weight_decay': 0.0001,
        'momentum': 0.9,
        'epochs': 60,
        'imgsize': (224, 244),
        # 'arch': args.arch,
        'version': 1,
        'make_loss': 0.2,
        'model_loss': 0.2,
        'submodel_loss':0.2,
        'finetune': False,
        'dataset':1,
        'split':'hard',
        # 'path': args.path
    }
    print("------")
    pp.pprint(config)
    print("------")
    if(config['dataset']==1):
        num_classes = 196 # Stanford
        num_makes = num_models = num_submodels = 0
    elif(config['dataset']==2):
        num_classes = 107 #- BoxCar Hard split['hard']['types_mapping']
        num_makes = 1
        num_models = 1
        num_submodels = 1
    else:
        print("Incorrect dataset")
        exit(1)
    # Create model
    model = construct_model(config, num_classes,num_makes,num_models,num_submodels)

    # Finetune an existing model already trained
    # if config['finetune']:
    #     load_weight(model, config['path'], device)

    #Add to multiple paramaters
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    # Addes model to GPU
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(),
                          lr=config['lr'],
                          momentum=config['momentum'],  # TODO what is this
                          weight_decay=config['weight_decay'])  # TODO what is this

    # Change the learning reate at 100/150 milestones(epochs). Decrease by 10*
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  [100, 150],
                                                  gamma=0.1) 



    # Set up data
    train_loader, test_loader = prepare_loader(config)
    
    best_acc = 0
    res = []

    if config['version'] == 1:
        train_fn = train_v1
        test_fn = test
    elif config['version'] == 2:
        train_fn = train_v2
        test_fn = test
    else:
        print(version, "is not a valid version number")
        exit(1)
    for ep in range(1, config['epochs'] + 1):
        trainres = train_fn(ep, model, optimizer, lr_scheduler, train_loader, device, config)
        valres = test_fn(model, test_loader, device, config)
        trainres.update(valres)

        if best_acc < valres['val_acc']:
            best_acc = valres['val_acc']
            torch.save(model.state_dict(), exp_dir + '/best.pth')
            trainres['overwritten']=1#Work out from excel which epoch the best model from
        else:
            trainres['overwritten']=0
        res.append(trainres)

    print(f'Best accuracy: {best_acc:.4f}')
    res = pd.DataFrame(res)
    res.to_csv(exp_dir + '/history.csv')


if __name__ == '__main__':
    main()
