import argparse
import json
import os
import pprint as pp

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from datasets import prepare_test_loader
from datasets.BoxCarsDataset import load_boxcar_class_names
from models import construct_model
from config import SAVE_FOLDER


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

    make_loss_meter = 0
    make_acc_meter = 0

    model_loss_meter = 0
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

            main_loss = F.cross_entropy(pred, target)
            make_loss = F.cross_entropy(make_pred, make_target)
            model_loss = F.cross_entropy(model_pred, model_target)

            loss = main_loss + config['make_loss'] * make_loss + config['model_loss'] * model_loss

            acc = pred.max(1)[1].eq(target).float().sum()
            make_acc = make_pred.max(1)[1].eq(make_target).float().sum()
            model_acc = model_pred.max(1)[1].eq(model_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            acc_meter += acc.item()

            make_loss_meter += make_loss.item()
            make_acc_meter += make_acc.item()
            
            model_loss_meter += model_loss.item()
            model_acc_meter += model_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Acc: {acc_meter / runcount:.4f} '

                  f'Make L: {make_loss_meter / runcount:.4f} '
                  f'Make A: {make_acc_meter / runcount:.4f} '

                  f'Model L: {model_loss_meter / runcount:.4f} '
                  f'Model A: {model_acc_meter / runcount:.4f} '
                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount
        acc_meter /= runcount

        make_loss_meter /= runcount
        make_acc_meter /= runcount

        model_loss_meter /= runcount
        model_acc_meter /= runcount

    print(f'Test Result: '
          f'Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} '
          f'Make L: {make_loss_meter:.4f} Make A: {make_acc_meter:.4f} '
          f'Model L: {model_loss_meter:.4f} Model A: {model_loss_meter:.4f} '
          f'({elapsed:.2f}s)')

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,

        'val_make_loss': make_loss_meter,
        'val_make_acc': make_acc_meter,

        'val_model_loss': model_loss_meter,
        'val_model_acc': model_acc_meter,

        'val_time': elapsed
    }

    return valres

def test_v3(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    acc_meter = 0

    make_loss_meter = 0
    make_acc_meter = 0

    model_loss_meter = 0
    model_acc_meter = 0

    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, model_target in test_loader:
            data = data.to(device)
            target = target.to(device)
            make_target = make_target.to(device)
            model_target = model_target.to(device)

            pred, make_pred, model_pred = model(data)

            main_loss = F.cross_entropy(pred, target)
            make_loss = F.cross_entropy(make_pred, make_target)
            model_loss = F.cross_entropy(model_pred, model_target)

            loss = main_loss + config['make_loss'] * make_loss + config['make_loss'] * model_loss

            acc = pred.max(1)[1].eq(target).float().sum()
            make_acc = make_pred.max(1)[1].eq(make_target).float().sum()
            model_acc = model_pred.max(1)[1].eq(model_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            acc_meter += acc.item()

            make_loss_meter += make_loss.item()
            make_acc_meter += make_acc.item()

            model_loss_meter += model_loss.item()
            model_acc_meter += model_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Acc: {acc_meter / runcount:.4f} '
                  f'Make L: {make_loss_meter / runcount:.4f} '
                  f'Make A: {make_acc_meter / runcount:.4f} '
                  f'Type L: {model_loss_meter / runcount:.4f} '
                  f'Type A: {model_acc_meter / runcount:.4f} '
                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount
        acc_meter /= runcount

        make_loss_meter /= runcount
        make_acc_meter /= runcount

        model_loss_meter /= runcount
        model_acc_meter /= runcount

    print(f'Test Result: '
          f'Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} '
          f'Make L: {make_loss_meter:.4f} Make A: {make_acc_meter:.4f} '
          f'Model L: {model_loss_meter:.4f} Model A: {model_loss_meter:.4f} '
          f'({elapsed:.2f}s)')


    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,

        'val_make_loss': make_loss_meter,
        'val_make_acc': make_acc_meter,

        'val_model_loss': model_loss_meter,
        'val_model_acc': model_acc_meter,

        'val_time': elapsed
    }

    return valres

def test_v4(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    acc_meter = 0
    make_loss_meter = 0
    make_acc_meter = 0
    model_loss_meter = 0
    model_acc_meter = 0
    submodel_loss_meter = 0
    submodel_acc_meter = 0

    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, model_target, submodel_target, generation_target in test_loader:
            data = data.to(device)
            target = target.to(device)
            make_target = make_target.to(device)
            model_target = model_target.to(device)
            submodel_target = submodel_target.to(device)

            pred, make_pred, model_pred,submodel_pred = model(data)

            main_loss = F.cross_entropy(pred, target)
            make_loss = F.cross_entropy(make_pred, make_target)
            model_loss = F.cross_entropy(model_pred, model_target)
            submodel_loss = F.cross_entropy(submodel_pred, submodel_target)

            loss = main_loss + config['make_loss'] * make_loss + config['model_loss'] * model_loss  + config['submodel_loss'] * submodel_loss

            acc = pred.max(1)[1].eq(target).float().sum()
            make_acc = make_pred.max(1)[1].eq(make_target).float().sum()
            model_acc = model_pred.max(1)[1].eq(model_target).float().sum()
            submodel_acc = submodel_pred.max(1)[1].eq(submodel_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            acc_meter += acc.item()

            make_loss_meter += make_loss.item()
            make_acc_meter += make_acc.item()

            model_loss_meter += model_loss.item()
            model_acc_meter += model_acc.item()

            submodel_loss_meter += submodel_loss.item()
            submodel_acc_meter += submodel_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Acc: {acc_meter / runcount:.4f} '
                  f'Make: {make_loss_meter / runcount:.4f} '
                  f'Make: {make_acc_meter / runcount:.4f} '

                  f'Model: {model_loss_meter / runcount:.4f} '
                  f'Model: {model_acc_meter / runcount:.4f} '

                  f'SubModel: {submodel_loss_meter / runcount:.4f} '
                  f'SubModel: {submodel_acc_meter / runcount:.4f} '

                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount
        acc_meter /= runcount

        make_loss_meter /= runcount
        make_acc_meter /= runcount

        model_loss_meter /= runcount
        model_acc_meter /= runcount

        submodel_loss_meter /= runcount
        submodel_acc_meter /= runcount


    print(f'Test Result: '
          f'Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} '
          f'Make L: {make_loss_meter:.4f} Make A: {make_acc_meter:.4f} '
          f'Model L: {model_loss_meter:.4f} Model A: {model_loss_meter:.4f} '
          f'Submodel L: {submodel_loss_meter:.4f} Submodel A: {submodel_loss_meter:.4f} '
          f'({elapsed:.2f}s)')


    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        
        'val_make_loss': make_loss_meter,
        'val_make_acc': make_acc_meter,

        'val_model_loss': model_loss_meter,
        'val_model_acc': model_acc_meter,

        'val_submodel_loss':submodel_loss_meter,
        'val_submodel_acc':submodel_acc_meter,
        'val_time': elapsed
    }

    return valres


# ---------------------------
# Classic multitask learning
# ---------------------------

#One model predicting make,model,submodel and generation seperatly
def test_v5(model, test_loader, device, config):
    model.eval()

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

    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, model_target, submodel_target, generation_target in test_loader:
            data = data.to(device)
            target = target.to(device)
            make_target = make_target.to(device)
            model_target = model_target.to(device)
            submodel_target = submodel_target.to(device)
            generation_target = generation_target.to(device)

            make_pred, model_pred,submodel_pred,generation_pred = model(data)

            make_loss = F.cross_entropy(make_pred, make_target)
            model_loss = F.cross_entropy(model_pred, model_target)
            submodel_loss = F.cross_entropy(submodel_pred, submodel_target)
            generation_loss = F.cross_entropy(generation_pred, generation_target)

            loss = config['make_loss'] * make_loss + config['model_loss'] * model_loss  + config['submodel_loss'] * submodel_loss + config['generation_loss'] * generation_loss

            make_acc = make_pred.max(1)[1].eq(make_target).float().sum()
            model_acc = model_pred.max(1)[1].eq(model_target).float().sum()
            submodel_acc = submodel_pred.max(1)[1].eq(submodel_target).float().sum()
            generation_acc = generation_pred.max(1)[1].eq(generation_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            # acc_meter += acc.item()

            make_loss_meter += make_loss.item()
            make_acc_meter += make_acc.item()

            model_loss_meter += model_loss.item()
            model_acc_meter += model_acc.item()

            submodel_loss_meter += submodel_loss.item()
            submodel_acc_meter += submodel_acc.item()
            
            generation_loss_meter += generation_loss.item()
            generation_acc_meter += generation_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                #   f'Acc: {acc_meter / runcount:.4f} '

                  f'Make L: {make_loss_meter / runcount:.4f} '
                  f'Make A: {make_acc_meter / runcount:.4f} '

                  f'Model L: {model_loss_meter / runcount:.4f} '
                  f'Model A: {model_acc_meter / runcount:.4f} '

                  f'SubModel L: {submodel_loss_meter / runcount:.4f} '
                  f'SubModel A: {submodel_acc_meter / runcount:.4f} '

                  f'Generation L: {generation_loss_meter / runcount:.4f} '
                  f'Generation A: {generation_acc_meter / runcount:.4f} '

                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount
        acc_meter /= runcount

        make_loss_meter /= runcount
        make_acc_meter /= runcount

        model_loss_meter /= runcount
        model_acc_meter /= runcount

        submodel_loss_meter /= runcount
        submodel_acc_meter /= runcount

        generation_loss_meter /= runcount
        generation_acc_meter /= runcount


    print(f'Test Result: '
          f'Loss: {loss_meter:.4f} Acc: {acc_meter:.4f} '
          f'Make L: {make_loss_meter:.4f} Make A: {make_acc_meter:.4f} '
          f'Model L: {model_loss_meter:.4f} Model A: {model_loss_meter:.4f} '
          f'Submodel L: {submodel_loss_meter:.4f} Submodel A: {submodel_loss_meter:.4f} '
          f'Generation L: {generation_loss_meter:.4f} Generation A: {generation_acc_meter:.4f} '
          f'({elapsed:.2f}s)')


    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,#Not used but kept to not affect output result layout
        
        'val_make_loss': make_loss_meter,
        'val_make_acc': make_acc_meter,

        'val_model_loss': model_loss_meter,
        'val_model_acc': model_acc_meter,

        'val_submodel_loss':submodel_loss_meter,
        'val_submodel_acc':submodel_acc_meter,

        'val_generation_loss':generation_loss_meter,
        'val_generation_acc':generation_acc_meter,


        'val_time': elapsed
    }

    return valres

#Model just predicts make
def test_v6(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    make_loss_meter = 0
    make_acc_meter = 0
    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, model_target, submodel_target, generation_target in test_loader:
            data = data.to(device)
            make_target = make_target.to(device)

            make_pred, model_pred,submodel_pred,generation_pred = model(data)

            make_loss = F.cross_entropy(make_pred, make_target)

            loss = config['make_loss'] * make_loss

            make_acc = make_pred.max(1)[1].eq(make_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            make_loss_meter += make_loss.item()
            make_acc_meter += make_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '

                  f'Make L: {make_loss_meter / runcount:.4f} '
                  f'Make A: {make_acc_meter / runcount:.4f} '

                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount

        make_loss_meter /= runcount
        make_acc_meter /= runcount


    print(f'Test Result: '
          f'Loss: {loss_meter:.4f} '
          f'Make L: {make_loss_meter:.4f} Make A: {make_acc_meter:.4f} '
          f'({elapsed:.2f}s)')


    valres = {
        'val_loss': loss_meter,
        'val_acc': -1,
        
        'val_make_loss': make_loss_meter,
        'val_make_acc': make_acc_meter,

        'val_model_loss': -1,
        'val_model_acc': -1,

        'val_submodel_loss':-1,
        'val_submodel_acc':-1,

        'val_generation_loss':-1,
        'val_generation_acc':-1,

        'val_time': elapsed
    }

    return valres

#Model just predicts model
def test_v7(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    model_loss_meter = 0
    model_acc_meter = 0
    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, model_target, submodel_target, generation_target in test_loader:
            data = data.to(device)
            model_target = model_target.to(device)

            make_pred, model_pred,submodel_pred,generation_pred = model(data)

            model_loss = F.cross_entropy(model_pred, model_target)

            loss = config['model_loss'] * model_loss

            model_acc = model_pred.max(1)[1].eq(model_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            model_loss_meter += model_loss.item()
            model_acc_meter += model_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Model L: {model_loss_meter / runcount:.4f} '
                  f'Model A: {model_acc_meter / runcount:.4f} '
                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount

        model_loss_meter /= runcount
        model_acc_meter /= runcount


    print(f'Test Result: '
          f'Loss: {loss_meter:.4f} '
          f'Model L: {model_loss_meter:.4f} Model A: {model_loss_meter:.4f} '
          f'({elapsed:.2f}s)')


    valres = {
        'val_loss': loss_meter,
        'val_acc': -1,#Not used but kept to not affect output result layout
        
        'val_make_loss': -1,
        'val_make_acc': -1,

        'val_model_loss': model_loss_meter,
        'val_model_acc': model_acc_meter,

        'val_submodel_loss':-1,
        'val_submodel_acc':-1,

        'val_generation_loss':-1,
        'val_generation_acc':-1,

        'val_time': elapsed
    }

    return valres

#Model just predicts submodel
def test_v8(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    submodel_loss_meter = 0
    submodel_acc_meter = 0
    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, model_target, submodel_target, generation_target in test_loader:
            data = data.to(device)
            submodel_target = submodel_target.to(device)

            make_pred, model_pred,submodel_pred,generation_pred = model(data)

            submodel_loss = F.cross_entropy(submodel_pred, submodel_target)

            loss = config['submodel_loss'] * submodel_loss

            submodel_acc = submodel_pred.max(1)[1].eq(submodel_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            submodel_loss_meter += submodel_loss.item()
            submodel_acc_meter += submodel_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'SubModel L: {submodel_loss_meter / runcount:.4f} '
                  f'SubModel A: {submodel_acc_meter / runcount:.4f} '
                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time

        loss_meter /= runcount
        submodel_loss_meter /= runcount
        submodel_acc_meter /= runcount

    print(f'Test Result: '
          f'Loss: {loss_meter:.4f} '
          f'Submodel L: {submodel_loss_meter:.4f} Submodel A: {submodel_loss_meter:.4f} '
          f'({elapsed:.2f}s)')


    valres = {
        'val_loss': loss_meter,
        'val_acc': -1,#Not used but kept to not affect output result layout
        
        'val_make_loss': -1,
        'val_make_acc': -1,

        'val_model_loss': -1,
        'val_model_acc': -1,

        'val_submodel_loss':submodel_loss_meter,
        'val_submodel_acc':submodel_acc_meter,

        'val_generation_loss':-1,
        'val_generation_acc':-1,


        'val_time': elapsed
    }

    return valres

#Model just predicts generation
def test_v9(model, test_loader, device, config):
    model.eval()

    loss_meter = 0
    generation_loss_meter = 0
    generation_acc_meter = 0

    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, model_target, submodel_target, generation_target in test_loader:
            data = data.to(device)
            generation_target = generation_target.to(device)

            make_pred, model_pred,submodel_pred,generation_pred = model(data)

            generation_loss = F.cross_entropy(generation_pred, generation_target)

            loss = config['generation_loss'] * generation_loss

            generation_acc = generation_pred.max(1)[1].eq(generation_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            generation_loss_meter += generation_loss.item()
            generation_acc_meter += generation_acc.item()

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print(f'[{i}/{len(test_loader)}]: '
                  f'Loss: {loss_meter / runcount:.4f} '
                  f'Generation L: {generation_loss_meter / runcount:.4f} '
                  f'Generation A: {generation_acc_meter / runcount:.4f} '

                  f'({elapsed:.2f}s)', end='\r')

        print()

        elapsed = time.time() - start_time
        
        loss_meter /= runcount
        generation_loss_meter /= runcount
        generation_acc_meter /= runcount


    print(f'Test Result: '
          f'Loss: {loss_meter:.4f}'
          f'Generation L: {generation_loss_meter:.4f} Generation A: {generation_acc_meter:.4f} '
          f'({elapsed:.2f}s)')


    valres = {
        'val_loss': loss_meter,
        'val_acc': -1,#Not used but kept to not affect output result layout
        
        'val_make_loss': -1,
        'val_make_acc': -1,

        'val_model_loss': -1,
        'val_model_acc': -1,

        'val_submodel_loss':-1,
        'val_submodel_acc':-1,

        'val_generation_loss':generation_loss_meter,
        'val_generation_acc':generation_acc_meter,


        'val_time': elapsed
    }

    return valres

# -------------------------------
# Classic multitask learning END
# -------------------------------


def load_weight(model, path, device):
    sd = torch.load(path,map_location=device)
    model.load_state_dict(sd)


def main(args):
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    config = {
        'model_id':args.model_id,
        'model_version':args.model_version,
        'dataset_version':args.dataset_version,
        'imgsize': (244,244),
        'boxcar_split':'hard',
        'test_batch_size':60,

        'make_loss': args.make_loss,
        'model_loss': args.model_loss,
        'submodel_loss':args.submodel_loss,  
        'generation_loss':args.generation_loss,    
    }

    pp.pprint(config)

    modelpath = os.path.join(SAVE_FOLDER, str(config['model_id']).zfill(3)+'_model.pth')

    multi_nums,test_loader = prepare_test_loader(config)

    model = construct_model(config, multi_nums['num_classes'],multi_nums['num_makes'],multi_nums['num_models'],multi_nums['num_submodels'],multi_nums['num_generations'])
    load_weight(model, modelpath, device)
    model = model.to(device)

    if config['model_version'] in [2]: 
        print("Test Version 2 for boxcars (Multitask learning - 2 features) ")
        test_fn = test_v2
    elif config['model_version'] in [9,10]:
        print("Test Version 4 for boxcars (Multitask learning - 3 features)")
        test_fn = test_v4
    elif config['model_version'] in [11]:
        print("Test Version 5 for boxcars (multitask learning with classifc ml format")
        test_fn = test_v5
    elif config['model_version'] in [8]:
        print("Test Version 3 for stanford (Multitask learning)")
        test_fn = test_v3
    else:
        print("Test Version 1 for normal models")
        test_fn = test_v1
    test_fn(model, test_loader, device, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script for Cars dataset')

    parser.add_argument('--model-id',default=15,type=int,required=True,
                        help='id to lined to previous model to fine tune. Required if it is a fine tune task')
    parser.add_argument('--model-version', default=2, type=int,required=True,
                        help='Classification version (default: 2)\n'
                             '1. Full Annotation only\n'
                             '2. Multitask Learning Cars Model + Make + Model + Submodel')
    parser.add_argument('--dataset-version', default=1, type=int, choices=[1,2],required=True,
                        help='Classification version (default: 1)\n'
                             '1. Stanford Dataset\n'
                             '2. BoxCar Dataset')

     # multi-task learning arg
    parser.add_argument('--make-loss', default=0.2, type=float,
                        help='loss$_{make}$ lambda')
    parser.add_argument('--model-loss', default=0.2, type=float,
                        help='loss$_{model}$ lambda')
    parser.add_argument('--submodel-loss', default=0.2, type=float,
                        help='loss$_{submodel}$ lambda')

    #Not used but added so I didn;t need to add conditions in bash file to remove them
    parser.add_argument('--epochs', default=150, type=int,
                        help='training epochs (default: 60)')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='training batch size (default: 32)')
    # optimizer arg
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='SGD learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', default=0.0001, type=float,
                        help='SGD weight decay (default: 0.0001)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--adam', default=False, action='store_true',
                        help='Use adam over SVG(SVG by default)')
    parser.add_argument('--generation-loss', default=0.2, type=float,
                        help='loss$_{generation}$ lambda')
    args = parser.parse_args()

    main(args)