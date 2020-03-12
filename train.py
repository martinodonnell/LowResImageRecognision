from datasets import prepare_loader
from models import construct_model
from trainTestUtil import get_train_test_methods, set_up_output_filepaths, get_output_filepaths
import argparse
import os
import pprint as pp
import time

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd


# -------------------------------
# Classic multitask learning END
# -------------------------------

def load_weight(model, path, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)


def load_weight_stan_boxcars(model, path, device):
    pretrained_dict = torch.load(path, map_location=device)
    pretrained_dict_ids = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    # Add features
    for i in pretrained_dict_ids:
        key = 'base.features.' + str(i)
        model.state_dict()[key + '.weight'].data.copy_(pretrained_dict[key + '.weight'])
        model.state_dict()[key + '.bias'].data.copy_(pretrained_dict[key + '.bias'])

    # #Add classifiers
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
        'train_test_version': args.train_test_version,
        'dataset_version': args.dataset_version,
        'boxcar_split': args.boxcar_split,
        'train_samples':int(args.train_samples),
        'finetune': args.finetune,
        'model_id': args.model_id,
        'finetune_stan_box': args.finetune_stan_box,

        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'adam': args.adam,

        'make_loss': args.make_loss,
        'model_loss': args.model_loss,
        'submodel_loss': args.submodel_loss,
        'generation_loss': args.generation_loss,
    }

    # Output so kelvin output file prints it
    pp.pprint(config)

    # Set up data loaders
    train_loader, test_loader, confusion_matrixes = prepare_loader(config)

    # Create model
    model = construct_model(config, config['num_classes'], config['num_makes'], config['num_models'],
                            config['num_submodels'], config['num_generations'])

    csv_history_filepath, model_best_filepath = get_output_filepaths(config)

    # Finetune an existing model already trained
    if config['finetune']:
        print("Loading existing model", )
        if config['finetune_stan_box']:
            load_weight_stan_boxcars(model, model_best_filepath, device)
        else:
            load_weight(model, model_best_filepath, device)

            # Add to multiple gpus if they are there
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Adds model to GPU
    model = model.to(device)

    # Set up output files and add header to csv file
    config, csv_history_filepath, model_best_filepath = set_up_output_filepaths(config)
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
    train_fn, test_fn = get_train_test_methods(config)

    best_acc = 0
    res = []
    for ep in range(1, config['epochs'] + 1):
        trainres = train_fn(ep, model, optimizer, train_loader, device, config)
        valres = test_fn(model, test_loader, device, config, None)
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

    print(f'Best accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and finetuning script for Cars classification task')

    # training arg
    parser.add_argument('--batch-size', default=32, type=int,
                        help='training batch size (default: 32)')
    parser.add_argument('--epochs', default=40, type=int,
                        help='training epochs (default: 60)')
    parser.add_argument('--imgsize', default=224, type=int,
                        help='Input image size (default: 224)')
    parser.add_argument('--model-version', default=1, type=int,
                        help='Classification version (default: 1)\n'
                             '1. Full Annotation only\n'
                             '2. Multitask Learning Cars Model + Make + Model + Submodel')
    parser.add_argument('--train-test-version', default=1, type=int,
                        help='Some models have more than one test_train setup giving different training and test '
                             'abilities)\n')
    parser.add_argument('--dataset-version', default=1, type=int, choices=[1, 2, 3, 4],
                        help='Classification version (default: 1)\n'
                             '1. Stanford Dataset\n'
                             '2. BoxCar Dataset\n'
                             '3. BoxCar Dataset with Augmentation')
    parser.add_argument('--boxcar-split', default='hard',
                        help='required if set dataset-version to 2 (default: hard)')
    parser.add_argument('--train-samples', default=1,type=int,
                        help='required if set dataset-version to 4 (default: 1). Determines how many training samples to train the model with during testing from boxcars')
    parser.add_argument('--finetune', default=False, action='store_true',
                        help='fine tune an existing model (default: False)')
    parser.add_argument('--finetune-stan-box', default=False, action='store_true',
                        help='Fix architecture for boxcars(default: False)')
    parser.add_argument('--model-id', default=32, type=int,
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
    parser.add_argument('--generation_loss', default=0.2, type=float,
                        help='loss$_{generation}$ lambda')

    args = parser.parse_args()
    main(args)
