from datasets import prepare_loader
from models import construct_model
from train_test_util import set_up_output_filepaths, get_output_filepaths,get_args,load_weight,load_weight_stanford,get_loss_function
from train_test import get_train_test_methods
import os
import pprint as pp
import time

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd




def main(config):
    # TODO what does this do
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Set up data loaders
    train_loader, test_loader, confusion_matrixes = prepare_loader(config)

    # Create model
    model = construct_model(config['model_version'], config['num_classes'], config['num_makes'], config['num_models'],
                            config['num_submodels'], config['num_generations'])
    
    csv_history_filepath, model_best_filepath = get_output_filepaths(config)

    # Finetune an existing model already trained
    if config['finetune']:
        _, fine_tune_model_path = get_output_filepaths(config['fine-tune-id'])
        print("finetune model ", fine_tune_model_path)
        if config['finetune_stan_box']:
            #Match the last layers of boxcars to stanford
            model.base.classifier[6] = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(4096, 196),
            )
            #Load weights
            load_weight_stanford(model, fine_tune_model_path, device)
            #Revert model to the same as before with random weights
            if(config['model_version'] is 7):
                #Change 
                model.base.classifier[6] = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(4096, config['num_classes']),
                )
            elif(config['model_version'] in [15,16,17,18,19,20]):
                #Remove final layer to allow auxillary layers to do all the work
                model.base.classifier[6] = nn.Sequential(
                    nn.Dropout(0.5),
                )                
        else:
            load_weight(model, fine_tune_model_path, device)

    # Add to multiple gpus if they are there
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Adds model to GPU
    model = model.to(device)
  
    # Set up output files and add header to csv file
    csv_history_filepath, model_best_filepath = set_up_output_filepaths(config['testing'],config['model_id'],config['train_test_version'])
    print(csv_history_filepath, model_best_filepath)

    # Decide on optimiser. Adam works best but keeping it here for later
    if config['adam']:
        optimizer = optim.Adam(model.parameters(),  # Contains link to learnable parameters
                               betas=(0.9, 0.999),  # Other parameters are other optimiser parameters
                               lr=config['lr'],
                               weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=config['lr'],
                              momentum=config['momentum'],  # TODO what is this
                              weight_decay=config['weight_decay'])  # TODO what is this

    # May need to use a different one. This one is not cutting it
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Get correct train/test methods
    train_fn, test_fn = get_train_test_methods(config['train_test_version'])

    #Get loss function
    loss_function = get_loss_function(config['loss-function'])
    

    best_acc = 0
    res = []
    for ep in range(1, config['epochs'] + 1):
        trainres = train_fn(ep, model, optimizer, train_loader, device, config,loss_function)
        valres = test_fn(model, test_loader, device, config, None,loss_function)
        trainres.update(valres)
        trainres['lr'] = optimizer.param_groups[0]['lr']
        lr_scheduler.step(trainres['val_loss'])

        if best_acc < valres['val_acc']:
            best_acc = valres['val_acc']
            torch.save(model.state_dict(), model_best_filepath)
            trainres['overwritten'] = 1
        else:
            trainres['overwritten'] = 0
        trainres['epoch'] = ep

        # This should save each result as we go along instead of at the end
        res = pd.DataFrame([trainres])
        res.to_csv(csv_history_filepath, mode='a', header=None)

    print("Best accuracy: {:,.4f}".format(best_acc))



if __name__ == '__main__':
    config = get_args()
    main(config)
