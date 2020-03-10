
import torch.nn.functional as F
import time
import torch

def update_confusion_matrix(matrix,pred,targets):
    preds = torch.argmax(pred, 1)
    for p, t in zip(preds, targets):
        matrix[p, t] += 1
        
def move_data_to_device(device,data,target,make_target,model_target,submodel_target,generation_target):
    data = data.to(device)
    target = target.to(device)
    make_target = make_target.to(device)
    model_target = model_target.to(device)
    submodel_target = submodel_target.to(device)
    generation_target = generation_target.to(device)
    
    return data,target,make_target,model_target,submodel_target,generation_target

def generate_fine_tune_metrics():

    metrics = {
        'loss_meter':0,
        'acc_meter':0,
        'make_loss_meter':0,
        'make_acc_meter':0,
        'model_loss_meter':0,
        'model_acc_meter':0,
        'submodel_loss_meter':0,
        'submodel_acc_meter':0,
        'generation_loss_meter':0,
        'generation_acc_meter':0
    }

    return metrics


def cal_loss(make,model,submodel,generation,config,metrics):
    make_loss = F.cross_entropy(make[0], make[1])
    model_loss = F.cross_entropy(model[0], model[1])
    submodel_loss = F.cross_entropy(submodel[0], submodel[1])
    generation_loss = F.cross_entropy(generation[0], generation[1])

    make_acc = make[0].max(1)[1].eq(make[1]).float().mean()
    model_acc = model[0].max(1)[1].eq(model[1]).float().mean()
    submodel_acc = submodel[0].max(1)[1].eq(submodel[1]).float().mean()
    generation_acc = generation[0].max(1)[1].eq(generation[1]).float().mean()

    #Make
    metrics['make_loss_meter'] += make_loss.item()
    metrics['make_acc_meter'] += make_acc.item()

    #Model
    metrics['model_loss_meter'] += model_loss.item()
    metrics['model_acc_meter'] += model_acc.item()

    #Submodel
    metrics['submodel_loss_meter'] += submodel_loss.item()
    metrics['submodel_acc_meter'] += submodel_acc.item()

    #Generation
    metrics['generation_loss_meter'] += generation_loss.item()
    metrics['generation_acc_meter'] += generation_acc.item()


    loss = config['make_loss'] * make_loss + config['model_loss'] * model_loss + config['submodel_loss'] * submodel_loss + config['generation_loss'] * generation_loss

    metrics['loss_meter']+=loss
    metrics['acc_meter']+= (metrics['make_acc_meter']+metrics['model_acc_meter']+metrics['submodel_acc_meter']+metrics['generation_acc_meter'])/4
    return loss

def print_single_ep_values(ep,i,load_size,elapsed,metrics):
    print(f'Epoch {ep:03d} [{i}/{load_size}]: '

        f'Loss: {metrics["loss_meter"] / i:.4f} '
        f'Acc: {metrics["acc_meter"] / i:.4f} '

        f'Make L: {metrics["make_loss_meter"] / i:.4f} '
        f'Make A: {metrics["make_acc_meter"] / i:.4f} '

        f'Model L: {metrics["model_loss_meter"] / i:.4f} '
        f'Model A: {metrics["model_acc_meter"] / i:.4f} '

        f'SubModel L: {metrics["submodel_loss_meter"] / i:.4f} '
        f'SubModel A: {metrics["submodel_acc_meter"] / i:.4f} '

        f'Generation L: {metrics["generation_loss_meter"] / i:.4f} '
        f'Generation A: {metrics["generation_acc_meter"] / i:.4f} '

        f'({elapsed:.2f}s)', end='\r')

def get_average_loss_accc(metrics,train_load_size):
    for key in metrics:
        metrics[key] /= train_load_size

def save_metrics_to_dict(metrics,elapsed,types):
    saves = {
        types+'_loss': metrics['loss_meter'],
        types+'_acc': metrics['acc_meter'],

        types+'_make_loss': metrics['make_loss_meter'],
        types+'_make_acc':metrics['make_acc_meter'],

        types+'_model_loss': metrics['model_loss_meter'],
        types+'_model_acc':metrics['model_acc_meter'],

        types+'_submodel_loss':metrics['submodel_loss_meter'],
        types+'submodel_acc':metrics['submodel_acc_meter'],

        types+'generation_loss':metrics['generation_loss_meter'],
        types+'generation_acc':metrics['generation_acc_meter'],

        types+'_time': elapsed
    }

    return saves


#Predicit each feature for label and backpropogate with combined loss
def train_v5(ep, model, optimizer, train_loader, device, config):
    model.train()

    #Get dictionary of metrics
    metrics = generate_fine_tune_metrics()

    i = 0
    start_time = time.time()
    elapsed = 0

    for data, target,make_target, model_target,submodel_target,generation_target in train_loader:

        #Add data to gpu
        data,target,make_target,model_target,submodel_target,generation_target = move_data_to_device(device,data,target,make_target,model_target,submodel_target,generation_target)

        optimizer.zero_grad()

        #Make predictions
        make_pred, model_pred,submodel_pred,generation_pred = model(data)
        
        #Calculate loss and add to metrics
        loss = cal_loss((make_pred, make_target),(model_pred, model_target),(submodel_pred, submodel_target),(generation_pred, generation_target),
                         config,metrics)

        #Back propogation
        loss.backward()
        optimizer.step()

        i += 1
        elapsed = time.time() - start_time
        print("test")
        print_single_ep_values(ep,i,len(train_loader),elapsed,metrics)

    print()

    get_average_loss_accc(metrics,len(train_loader))

    trainres = save_metrics_to_dict(metrics,elapsed,'train')

    return trainres


#One model predicting make,model,submodel and generation seperatly
def test_v5(model, test_loader, device, config,confusion_matrix):
    model.eval()

    #Get dictionary of metrics
    metrics = generate_fine_tune_metrics()
    runcount = 0

    i = 0

    with torch.no_grad():
        start_time = time.time()
        for data, target, make_target, model_target, submodel_target, generation_target in test_loader:
            #Add data to gpu
            data,target,make_target,model_target,submodel_target,generation_target = move_data_to_device(device,data,target,make_target,model_target,submodel_target,generation_target)

            make_pred, model_pred,submodel_pred,generation_pred = model(data)

            #Calculate loss and add to metrics
            cal_loss((make_pred, make_target),(model_pred, model_target),(submodel_pred, submodel_target),(generation_pred, generation_target),
                         config,metrics)

            if (not confusion_matrix==None):
                update_confusion_matrix(confusion_matrix['make'],make_pred,make_target)
                update_confusion_matrix(confusion_matrix['model'],model_pred,model_target)
                update_confusion_matrix(confusion_matrix['submodel'],submodel_pred,submodel_target)
                update_confusion_matrix(confusion_matrix['generation'],generation_pred,generation_target)

            runcount += data.size(0)
            i += 1
            elapsed = time.time() - start_time

            print_single_ep_values(1,i,len(test_loader),elapsed,metrics)


        print()

        get_average_loss_accc(metrics,len(test_loader))

        valres = save_metrics_to_dict(metrics,elapsed,'val')
    return valres