import torch.nn.functional as F
import time
import torch
from trainTest.confusion_matrix import update_confusion_matrix


#Predicit each feature for label and backpropogate with combined loss
def train_v6(ep, model, optimizer, train_loader, device, config, loss_function):
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

    for data, main_target,make_target,model_target,submodel_target,generation_target in train_loader:
        data = data.to(device)
        main_target = main_target.to(device)
        make_target = make_target.to(device)
        model_target = model_target.to(device)
        submodel_target = submodel_target.to(device)
        generation_target = generation_target.to(device)

        optimizer.zero_grad()

        main_pred, make_pred, model_pred,submodel_pred,generation_pred = model(data)
        
        #Calucalate individual loss for part labels
        main_loss = loss_function(main_pred, main_target)
        make_loss = loss_function(make_pred, make_target)
        model_loss = loss_function(model_pred, model_target)
        submodel_loss = loss_function(submodel_pred, submodel_target)
        generation_loss = loss_function(generation_pred, generation_target)
        
        loss = main_loss * config['main_loss'] + config['make_loss'] * make_loss + config['model_loss'] * model_loss + config['submodel_loss'] * submodel_loss + config['generation_loss'] * generation_loss
        loss.backward()

        optimizer.step()
        
        main_acc = torch.max(main_pred,1).indices.eq(main_target).float().mean()
        make_acc = torch.max(make_pred,1).indices.eq(make_target).float().mean()
        model_acc = torch.max(model_pred,1).indices.eq(model_target).float().mean()
        submodel_acc = torch.max(submodel_pred,1).indices.eq(submodel_target).float().mean()
        generation_acc = torch.max(generation_pred,1).indices.eq(generation_target).float().mean()

        #Main
        loss_meter += loss.item()
        acc_meter += main_acc.item()

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
              f'Acc: {acc_meter / i:.4f} '

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
        'train_acc': acc_meter,

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



#One model predicting make,model,submodel and generation seperatly
def test_v6(model, test_loader, device, config,confusion_matrix,loss_function):
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
        for data, main_target, make_target, model_target, submodel_target, generation_target in test_loader:
            data = data.to(device)
            main_target = main_target.to(device)
            make_target = make_target.to(device)
            model_target = model_target.to(device)
            submodel_target = submodel_target.to(device)
            generation_target = generation_target.to(device)

            main_pred, make_pred, model_pred,submodel_pred,generation_pred = model(data)

            main_loss = loss_function(main_pred, main_target)
            make_loss = loss_function(make_pred, make_target)
            model_loss = loss_function(model_pred, model_target)
            submodel_loss = loss_function(submodel_pred, submodel_target)
            generation_loss = loss_function(generation_pred, generation_target)
            
            loss = main_loss * config['main_loss'] + config['make_loss'] * make_loss + config['model_loss'] * model_loss + config['submodel_loss'] * submodel_loss + config['generation_loss'] * generation_loss

            main_acc = torch.max(main_pred,1).indices.eq(main_target).float().sum()
            make_acc = torch.max(make_pred,1).indices.eq(make_target).float().sum()
            model_acc = torch.max(model_pred,1).indices.eq(model_target).float().sum()
            submodel_acc = torch.max(submodel_pred,1).indices.eq(submodel_target).float().sum()
            generation_acc = torch.max(generation_pred,1).indices.eq(generation_target).float().sum()

            if (not confusion_matrix==None):
                update_confusion_matrix(confusion_matrix['total'],main_pred,main_target)
                update_confusion_matrix(confusion_matrix['make'],make_pred,make_target)
                update_confusion_matrix(confusion_matrix['model'],model_pred,model_target)
                update_confusion_matrix(confusion_matrix['submodel'],submodel_pred,submodel_target)
                update_confusion_matrix(confusion_matrix['generation'],generation_pred,generation_target)

            loss_meter += loss.item() * data.size(0)
            acc_meter += main_acc.item()

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
                  f'Acc: {acc_meter / runcount:.4f} '

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
          f'Model L: {model_loss_meter:.4f} Model A: {model_acc_meter:.4f} '
          f'Submodel L: {submodel_loss_meter:.4f} Submodel A: {submodel_acc_meter:.4f} '
          f'Generation L: {generation_loss_meter:.4f} Generation A: {generation_acc_meter:.4f} '
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

        'val_generation_loss':generation_loss_meter,
        'val_generation_acc':generation_acc_meter,


        'val_time': elapsed
    }

    return valres
