import torch
from train_test.confusion_matrix import update_confusion_matrix
import time

def train_v3(ep, model, optimizer, train_loader, device, config,loss_function):
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
        
        main_loss = loss_function(pred, target)
        make_loss = loss_function(make_pred, make_target)
        model_loss = loss_function(model_pred, model_target)

        loss = main_loss + config['make_loss'] * make_loss + config['make_loss'] * model_loss
        loss.backward()

        optimizer.step()
        
        acc = torch.max(pred,1).indices.eq(target).float().mean()
        make_acc = torch.max(make_pred,1).indices.eq(make_target).float().mean()
        model_acc = torch.max(model_pred,1).indices.eq(model_target).float().mean()


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


def test_v3(model, test_loader, device, config,confusion_matrix,loss_function):
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

            main_loss = loss_function(pred, target)
            make_loss = loss_function(make_pred, make_target)
            model_loss = loss_function(model_pred, model_target)

            loss = main_loss + config['make_loss'] * make_loss + config['make_loss'] * model_loss

            acc = torch.max(pred,1).indices.eq(target).float().sum()
            make_acc = torch.max(make_pred,1).indices.eq(make_target).float().sum()
            model_acc = torch.max(model_pred,1).indices.eq(model_target).float().sum()
            if (not confusion_matrix==None):
                update_confusion_matrix(confusion_matrix['total'],pred,target)
                update_confusion_matrix(confusion_matrix['make'],make_pred,make_target)
                update_confusion_matrix(confusion_matrix['model'],model_pred,model_target)

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
          f'Model L: {model_loss_meter:.4f} Model A: {model_acc_meter:.4f} '
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


