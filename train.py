from datasets import prepare_loader
from models import construct_model
from config import SAVE_FOLDER

import argparse
import os
import pprint as pp
import time

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd

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

# def train_v2(ep, model, optimizer, train_loader, device, config):

#     print("---------Training-------")

#     model.train() # Set model to training mode

#     loss_meter = 0
#     acc_meter = 0
#     make_acc_meter = 0
#     model_acc_meter = 0 
#     submodel_acc_meter = 0
#     i = 0

#     start_time = time.time()
#     elapsed = 0
#     for data, target,make_target,model_target,submodel_target,generation_target in train_loader:
#         data = data.to(device)
#         target = target.to(device)
#         make_target = make_target.to(device)
#         model_target = model_target.to(device)
#         submodel_target = submodel_target.to(device)
#         # generation_target = generation_target.to(device) #TODO ADD THIS ONE TOO 

#          # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         pred, make_pred, model_pred,submodel_pred = model(data)

#         loss_main = F.cross_entropy(pred, target)
#         loss_make = F.cross_entropy(make_pred, make_target)
#         loss_model = F.cross_entropy(model_pred, model_target)
#         loss_submodel = F.cross_entropy(submodel_pred, submodel_target)

#         loss = loss_main + (config['make_loss'] * loss_make) + (config['model_loss'] * loss_model) + (config['submodel_loss'] * loss_submodel)
#         loss.backward()
#         optimizer.step()

#         acc = pred.max(1)[1].eq(target).float().mean()
#         make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
#         model_acc = model_pred.max(1)[1].eq(model_target).float().mean()
#         submodel_acc = submodel_pred.max(1)[1].eq(submodel_target).float().mean()

#         loss_meter += loss.item()
#         acc_meter += acc.item()
#         make_acc_meter += make_acc.item()
#         model_acc_meter += model_acc.item()
#         submodel_acc_meter += submodel_acc.item()


#         i += 1
#         elapsed = time.time() - start_time
        
#     #Moved this out of for as I don't watch it all the time and will speed up performace
#     print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
#             f'Loss: {loss_meter / i:.4f} '
#             f'Acc: {acc_meter / i:.4f} ({elapsed:.2f}s) '
#             f'Make: {make_acc_meter / i:.4f} '
#             f'model: {model_acc_meter / i:.4f} '
#             f'Subtype: {submodel_acc_meter / i:.4f} '
#             ,end='\r')

#     print()
#     loss_meter /= len(train_loader)
#     acc_meter /= len(train_loader)
#     make_acc_meter /= len(train_loader)
#     model_acc_meter /= len(train_loader)
#     submodel_acc_meter /= len(train_loader)


#     trainres = {
#         'train_loss': loss_meter,
#         'train_acc': acc_meter,
#         'train_make_acc': make_acc_meter,
#         'train_model_acc': model_acc_meter,
#         'train_submodel_acc': submodel_acc_meter,
#         'train_time': elapsed,
#     }

#     return trainres


def train_v2(ep, model, optimizer, train_loader, device, config):
    model.train()

    loss_meter = 0
    acc_meter = 0
    make_acc_meter = 0
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
        
        loss_main = F.cross_entropy(pred, target)
        loss_make = F.cross_entropy(make_pred, make_target)
        loss_type = F.cross_entropy(model_pred, model_target)

        loss = loss_main + config['make_loss'] * loss_make + config['modelloss'] * loss_type
        loss.backward()

        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        model_acc = model_pred.max(1)[1].eq(model_target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        make_acc_meter += make_acc.item()
        model_acc_meter += model_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} '
              f'Make: {make_acc_meter / i:.4f} '
              f'Type: {model_acc_meter / i:.4f} '
              f'({elapsed:.2f}s)', end='\r')

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)
    model_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_make_acc': make_acc_meter,
        'train_model_acc': model_acc_meter,
        'train_time': elapsed
    }

    return trainres

def train_v3(ep, model, optimizer, train_loader, device, config):
    model.train()

    loss_meter = 0
    acc_meter = 0
    make_acc_meter = 0
    type_acc_meter = 0

    i = 0

    start_time = time.time()
    elapsed = 0

    for data, target, make_target, type_target in train_loader:
        data = data.to(device)
        target = target.to(device)
        make_target = make_target.to(device)
        type_target = type_target.to(device)

        optimizer.zero_grad()

        pred, make_pred, type_pred = model(data)
        
        loss_main = F.cross_entropy(pred, target)
        loss_make = F.cross_entropy(make_pred, make_target)
        loss_type = F.cross_entropy(type_pred, type_target)

        loss = loss_main + config['make_loss'] * loss_make + config['make_loss'] * loss_type
        loss.backward()

        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        type_acc = type_pred.max(1)[1].eq(type_target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        make_acc_meter += make_acc.item()
        type_acc_meter += type_acc.item()

        i += 1
        elapsed = time.time() - start_time

        print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} '
              f'Make: {make_acc_meter / i:.4f} '
              f'Type: {type_acc_meter / i:.4f} '
              f'({elapsed:.2f}s)', end='\r')

    print()
    loss_meter /= len(train_loader)
    acc_meter /= len(train_loader)
    make_acc_meter /= len(train_loader)
    type_acc_meter /= len(train_loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_make_acc': make_acc_meter,
        'train_type_acc': type_acc_meter,
        'train_time': elapsed
    }

    return trainres

def test_v1(model, test_loader, device, config):
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

def test_v2(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    acc_meter = 0
    make_acc_meter = 0
    model_acc_meter = 0
    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, model_target, submodel_target, generation_target in test_loader:
            data = data.to(device)
            target = target.to(device)
            make_target = make_target.to(device)
            model_target = model_target.to(device)

            pred, make_pred, model_pred = model(data)

            loss_main = F.cross_entropy(pred, target)
            loss_make = F.cross_entropy(make_pred, make_target)
            loss_type = F.cross_entropy(model_pred, model_target)

            loss = loss_main + config['make_loss'] * loss_make + config['model_loss'] * loss_type

            acc = pred.max(1)[1].eq(target).float().sum()
            make_acc = make_pred.max(1)[1].eq(make_target).float().sum()
            model_acc = model_pred.max(1)[1].eq(model_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            acc_meter += acc.item()
            make_acc_meter += make_acc.item()
            model_acc_meter += model_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Acc: {acc_meter / runcount:.4f} '
                  f'Make: {make_acc_meter / runcount:.4f} '
                  f'Type: {model_acc_meter / runcount:.4f} '
                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount
        acc_meter /= runcount
        make_acc_meter /= runcount
        model_acc_meter /= runcount

    print(f'Test Result: Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} ({elapsed:.2f}s)')

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        'val_make_acc': make_acc_meter,
        'val_type_acc': type_acc_meter,
        'val_time': elapsed
    }

    return valres
    model.eval()

    loss_meter = 0
    acc_meter = 0
    make_acc_meter = 0
    model_acc_meter = 0 
    submodel_acc_meter = 0

    runcount = 0
    elapsed = 0
   

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target,make_target,model_target,submodel_target,generation_target in test_loader:
            data = data.to(device)
            target = target.to(device)
            data = data.to(device)
            target = target.to(device)
            make_target = make_target.to(device)
            model_target = model_target.to(device)
            submodel_target = submodel_target.to(device)
            # generation_target = generation_target.to(device) #TODO ADD THIS ONE TOO 


            pred, make_pred, model_pred,submodel_pred = model(data)

            loss_main = F.cross_entropy(pred, target)
            loss_make = F.cross_entropy(make_pred, make_target)
            loss_model = F.cross_entropy(model_pred, model_target)
            loss_submodel = F.cross_entropy(submodel_pred, submodel_target)

            acc = pred.max(1)[1].eq(target).float().mean()
            make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
            model_acc = model_pred.max(1)[1].eq(model_target).float().mean()
            submodel_acc = submodel_pred.max(1)[1].eq(submodel_target).float().mean()

            loss_meter += (loss_main + (config['make_loss'] * loss_make) + (config['model_loss'] * loss_model) + (config['submodel_loss'] * loss_submodel))* data.size(0)
            acc_meter += acc.item()
            make_acc_meter += make_acc.item()
            model_acc_meter += model_acc.item()
            submodel_acc_meter += submodel_acc.item()

            i += 1
            elapsed = time.time() - start_time
            runcount += data.size(0)

            print(f'[{i}/{len(test_loader)}]: '
                f'Loss: {loss_meter / runcount:.4f} '
                f'Acc: {acc_meter / runcount:.4f} ({elapsed:.2f}s)'
                f'Make: {make_acc_meter / i:.4f} '
                f'model: {model_acc_meter / i:.4f} '
                f'Subtype: {submodel_acc_meter / i:.4f} '
                , end='\r')

        print()

        loss_meter /= runcount
        acc_meter /= runcount

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        'train_make_acc': make_acc_meter,
        'train_model_acc': model_acc_meter,
        'train_submodel_acc': submodel_acc_meter,
        'val_time': elapsed,
    }

    # print(f'Test Result: Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} ({elapsed:.2f}s)')# printed twice

    return valres

def test_v3(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    acc_meter = 0
    make_acc_meter = 0
    type_acc_meter = 0
    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, type_target in test_loader:
            data = data.to(device)
            target = target.to(device)
            make_target = make_target.to(device)
            type_target = type_target.to(device)

            pred, make_pred, type_pred = model(data)

            loss_main = F.cross_entropy(pred, target)
            loss_make = F.cross_entropy(make_pred, make_target)
            loss_type = F.cross_entropy(type_pred, type_target)

            loss = loss_main + config['make_loss'] * loss_make + config['make_loss'] * loss_type

            acc = pred.max(1)[1].eq(target).float().sum()
            make_acc = make_pred.max(1)[1].eq(make_target).float().sum()
            type_acc = type_pred.max(1)[1].eq(type_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            acc_meter += acc.item()
            make_acc_meter += make_acc.item()
            type_acc_meter += type_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Acc: {acc_meter / runcount:.4f} '
                  f'Make: {make_acc_meter / runcount:.4f} '
                  f'Type: {type_acc_meter / runcount:.4f} '
                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount
        acc_meter /= runcount
        make_acc_meter /= runcount
        type_acc_meter /= runcount

    print(f'Test Result: Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} ({elapsed:.2f}s)')

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        'val_make_acc': make_acc_meter,
        'val_type_acc': type_acc_meter,
        'val_time': elapsed
    }

    return valres

def get_output_filepaths(id):
    id_string = str(id).zfill(3)
    csv_history_filepath = os.path.join(SAVE_FOLDER, id_string+'_history.csv')
    model_best_filepath  = os.path.join(SAVE_FOLDER, id_string+'_model.pth')

    return csv_history_filepath,model_best_filepath

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

def main(args):
    # TODO what does this do
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # Set up config
    config = {
        'batch_size': args.batch_size,
        'test_batch_size': args.batch_size,
        'epochs': args.epochs,
        'imgsize': (args.imgsize, args.imgsize),
        'model_version': args.model_version,
        'dataset_version':args.dataset_version,
        'boxcar_split':args.boxcar_split,
        'finetune': args.finetune,
        'model_id':args.model_id,
        'finetune_stan_box':args.finetune_stan_box,

        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'adam':args.adam,
        
        'make_loss': args.make_loss,
        'model_loss': args.model_loss,
        'submodel_loss':args.submodel_loss,        
    }


    # Set up data loaders
    multi_nums, train_loader, test_loader = prepare_loader(config)
    #Save to log file on Kelvin
    pp.pprint(config)

    #Set up name for output files
    csv_history_filepath,model_best_filepath  = get_output_filepaths(config['model_id'])

    # Create model
    model = construct_model(config, multi_nums['num_classes'],multi_nums['num_makes'],multi_nums['num_models'],multi_nums['num_submodels'])

     # Finetune an existing model already trained
    if config['finetune']:
        print("Loading existing model", )
        if config['finetune_stan_box']:
            load_weight_stan_boxcars(model, model_best_filepath, device)  
        else:
            load_weight(model, model_best_filepath, device)  


    #Add to multiple gpus if they are there
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    # Addes model to GPU
    model = model.to(device)

    #Check if file exists. If so increment the id and try again until a new one is generated. Will create a new file if finetuning to 
    # ensure results are not lost. May have changes the parameters and want to keep them. Will know from the logs
    while(os.path.isfile(csv_history_filepath)):
        config['model_id'] = config['model_id']+1
        csv_history_filepath,model_best_filepath  = get_output_filepaths(config['model_id'])
    print("Current ID:",config['model_id'])

    #Set up blank csv in save folder
    df = pd.DataFrame(columns=['train_loss','train_acc','train_time','val_loss','val_acc','val_time','lr','overwritten','epoch'])
    df.to_csv(csv_history_filepath)    
    
    if(config['adam']):
        optimizer = optim.Adam(model.parameters(),#Contains link to learnable parameters
                                betas=(0.9, 0.999), #Other parameters are other optimiser parameters
                                lr = config['lr'], 
                                weight_decay = config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(),
                          lr=config['lr'],
                          momentum=config['momentum'],  # TODO what is this
                          weight_decay=config['weight_decay'])  # TODO what is this

   


    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')    
    
    best_acc = 0
    res = []
       
    if config['model_version'] in [2,8]:
        if(config['dataset_version']==1):
            print("Version 3")
            train_fn = train_v3
            test_fn = test_v3
        else:        
            print("Version 2")
            train_fn = train_v2
            test_fn = test_v2
    else:
        print("Version 1")
        train_fn = train_v1
        test_fn = test_v1


    for ep in range(1, config['epochs'] + 1):
        trainres = train_fn(ep, model, optimizer, train_loader, device, config)
        valres = test_fn(model, test_loader, device, config)
        trainres.update(valres)
        trainres['lr'] = optimizer.param_groups[0]['lr']
        lr_scheduler.step(trainres['val_loss'])


        if best_acc < valres['val_acc']:
            best_acc = valres['val_acc']
            torch.save(model.state_dict(), model_best_filepath)
            trainres['overwritten']=1#Work out from excel which epoch the best model from
        else:
            trainres['overwritten']=0
        trainres['epoch'] = ep
        
        #This should save each result as we go along instead of at the end
        res = pd.DataFrame([trainres])
        res.to_csv(csv_history_filepath, mode='a',header=None)

    print(f'Best accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and finetuning script for Cars classification task')

    # training arg
    parser.add_argument('--batch-size', default=32, type=int,
                        help='training batch size (default: 32)')
    parser.add_argument('--epochs', default=150, type=int,
                        help='training epochs (default: 60)')
    parser.add_argument('--imgsize', default=224, type=int,
                        help='Input image size (default: 224)')
    parser.add_argument('--model-version', default=1, type=int,
                        help='Classification version (default: 1)\n'
                             '1. Full Annotation only\n'
                             '2. Multitask Learning Cars Model + Make + Model + Submodel')
    parser.add_argument('--dataset-version', default=1, type=int, choices=[1,2],
                        help='Classification version (default: 1)\n'
                             '1. Stanford Dataset\n'
                             '2. BoxCar Dataset')
    parser.add_argument('--boxcar-split',default='hard',
                        help='required if set dataset-version to 2(default: hard)')
    parser.add_argument('--finetune', default=False, action='store_true',
                        help='fine tune an existing model (default: False)')
    parser.add_argument('--finetune-stan-box', default=False, action='store_true',
                        help='Fix arhetecture for boxcars(default: False)')
    parser.add_argument('--model-id',default=15,type=int,
                        help='id to lined to previous model to fine tune. Required if it is a fine tune task')

    # optimizer arg
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='SGD learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', default=0.0001, type=float,
                        help='SGD weight decay (default: 0.0001)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--adam', default=False, action='store_true',
                        help='Use adam over SVG(SVG by default)')

    # multi-task learning arg
    parser.add_argument('--make-loss', default=0.2, type=float,
                        help='loss$_{make}$ lambda')
    parser.add_argument('--model-loss', default=0.2, type=float,
                        help='loss$_{model}$ lambda')
    parser.add_argument('--submodel-loss', default=0.2, type=float,
                        help='loss$_{submodel}$ lambda')

    args = parser.parse_args()
    main(args)
