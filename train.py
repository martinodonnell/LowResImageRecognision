from datasets import load_class_names, prepare_loader
from models import construct_model
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import pandas as pd
import pprint as pp

def train(ep, model, optimizer, lr_scheduler, train_loader, device, config):

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
        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} ({elapsed:.2f}s)', end='\r')

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_time': elapsed
    }

    return trainres


def test(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    acc_meter = 0
    runcount = 0
    elapsed = 0
    
    f1_meter = 0
    percision_meter =0
    recall_meter = 0
   

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

            f1_meter += f1_score(target, pred, average="samples")
            percision_meter += precision_score(target, pred, average="samples")
            recall_meter +=  recall_score(target, pred, average="samples")

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Acc: {acc_meter / runcount:.4f} ({elapsed:.2f}s)'
                  f'Per: {percision_meter / runcount:.4f}'
                  f'Rec: {recall_meter / runcount:.4f}'
                  f' F1: {f1_meter / runcount:.4f}', end='\r')

        print()

        loss_meter /= runcount
        acc_meter /= runcount

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        'val_time': elapsed,
        'val_percision':percision_meter,
        'val_recall':recall_meter,
        'val_f1':f1_meter,
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
        'lr': 0.01,
        'weight_decay': 0.0001,
        'momentum': 0.9,
        'epochs': 60,
        'imgsize': (224, 244),
        # 'arch': args.arch,
        # 'version': args.version,
        # 'make_loss': args.make_loss,
        # 'type_loss': args.type_loss,
        'finetune': False,
        'dataset':2,
        'split':'hard',
        # 'path': args.path
    }
    print("------")
    pp.pprint(config)
    print("------")
    if(config['dataset']==1):
        num_classes = 196 # Stanford
    elif(config['dataset']==2):
        num_classes = 107 #- BoxCar Hard split['hard']['types_mapping']
    else:
        print("Incorrect dataset")
        exit(1)
    # Create model
    model = construct_model('VGG', num_classes)

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

    for ep in range(1, config['epochs'] + 1):
        trainres = train(ep, model, optimizer, lr_scheduler, train_loader, device, config)
        valres = test(model, test_loader, device, config)
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
