import argparse
import json
import os
import pprint as pp

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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
