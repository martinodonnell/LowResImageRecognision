import os
from config import SAVE_FOLDER
import pandas as pd
import pprint as pp
import argparse
import torch

def get_output_filepaths(id):
    id_string = str(id).zfill(3)
    csv_history_filepath = os.path.join(SAVE_FOLDER, id_string + '_history.csv')
    model_best_filepath = os.path.join(SAVE_FOLDER, id_string + '_model.pth')

    return csv_history_filepath, model_best_filepath


def set_up_output_filepaths(config):
    
    #Stops the checcking for used ids. Will just give back testing file paths
    if(config['testing']):
        return config,"Testing_csv","Testing_Model"
    # Set up name for output files
    csv_history_filepath, model_best_filepath = get_output_filepaths(config['model_id'])

    # Check if file exists. If so increment the id and try again until a new one is generated. Will create a new file
    # if finetuning to ensure results are not lost. May have changes the parameters and want to keep them. Will know
    # from the logs
    if os.path.isfile(csv_history_filepath):
        print(config['model_id'],"id already in use (set_up_output_filepaths)")
        exit(1)
    print("Current ID:", config['model_id'])

    # Set up blank csv in save folder
    if config['train_test_version'] in [1,7]:
        df = pd.DataFrame(
            columns=['train_loss', 'train_acc', 'train_time', 'val_loss', 'val_acc', 'val_time', 'lr', 'overwritten',
                     'epoch'])
    elif config['train_test_version'] in [2,3]:
        # Multitask learning (2 features) Boxcars or #Multitask learning 2 features Stanford
        df = pd.DataFrame(columns=['train_loss', 'train_acc', 'train_make_loss', 'train_make_acc', 'train_model_loss',
                                   'train_model_acc', 'train_time', 'val_loss', 'val_acc', 'val_make_loss',
                                   'val_make_acc', 'val_model_loss', 'val_model_acc', 'val_time', 'lr', 'overwritten',
                                   'epoch'])
    elif config['train_test_version'] in [4]:  # Multitask learning (3 features) Boxcars
        df = pd.DataFrame(columns=['train_loss', 'train_acc', 'train_make_loss', 'train_make_acc', 'train_model_loss',
                                   'train_model_acc', 'train_submodel_loss', 'train_submodel_acc', 'train_time',
                                   'val_loss', 'val_acc', 'val_make_loss', 'val_make_acc', 'val_model_loss',
                                   'val_model_acc', 'val_submodel_loss', 'val_submodel_acc', 'val_time', 'lr',
                                   'overwritten', 'epoch'])
    elif config['train_test_version'] in [5,6]:  # Multitask learning with classifc ml format
        df = pd.DataFrame(columns=['train_loss', 'train_acc', 'train_make_loss', 'train_make_acc', 'train_model_loss',
                                   'train_model_acc', 'train_submodel_loss', 'train_submodel_acc',
                                   'train_generation_loss', 'train_generation_acc', 'train_time', 'val_loss', 'val_acc',
                                   'val_make_loss', 'val_make_acc', 'val_model_loss', 'val_model_acc',
                                   'val_submodel_loss', 'val_submodel_acc', 'val_generation_loss', 'val_generation_acc',
                                   'val_time', 'lr', 'overwritten', 'epoch'])
    else:
        print(config['train_test_version'], "is not a valid trainTest method(set_up_output_filepaths)")
        exit(1) 
       
    df.to_csv(csv_history_filepath)

    return config, csv_history_filepath, model_best_filepath

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


def load_weight_stan_boxcars2(model, path, device):
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

def get_args(): 
    parser = argparse.ArgumentParser(description='Training/Testing and finetuning script for Cars classification task')

    parser.add_argument('--batch-size', default=32, type=int,
                        help='training batch size (default: 32)')
    parser.add_argument('--epochs', default=40, type=int,
                        help='training epochs (default: 60)')
    parser.add_argument('--imgsize', default=224, type=int,
                        help='Input image size (default: 224)')
    parser.add_argument('--model-version', default=1, type=int,
                        help='Classification version (default: 1)\n'
                             '1. Full Annotation only\n'
                             '2. Multitask Learning Cars Model + Make + Model + Submodel\n'
                             'Loads more. See docs')
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
                        help='Fine tune stanfor dmodel with boxcars default: False)')    
    parser.add_argument('--ds-stanford', default=False, action='store_true',
                        help='Use the downsamples version of stanford dataset._ds.jpef as suffix')                      
    parser.add_argument('--fine-tune-id',type=int,
                        help='id to lined to previous model to fine tune. Required if it is a fine tune task')
    
    parser.add_argument('--model-id', type=int,
                        help='id to lined to previous model to fine tune. Required')
    
    parser.add_argument('--testing', default=False, action='store_true',
                        help='Will not output files if true')  
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
    parser.add_argument('--main-loss', default=1, type=float,
                        help='loss$_{main}$ lambda')
    parser.add_argument('--make-loss', default=0.2, type=float,
                        help='loss$_{make}$ lambda')
    parser.add_argument('--model-loss', default=0.2, type=float,
                        help='loss$_{model}$ lambda')
    parser.add_argument('--submodel-loss', default=0.2, type=float,
                        help='loss$_{submodel}$ lambda')
    parser.add_argument('--generation_loss', default=0.2, type=float,
                        help='loss$_{generation}$ lambda')

    args = parser.parse_args()

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
        'fine-tune-id':args.fine_tune_id,
        'ds-stanford':args.ds_stanford,
        
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'adam': args.adam,
        
        'main_loss': args.main_loss,
        'make_loss': args.make_loss,
        'model_loss': args.model_loss,
        'submodel_loss': args.submodel_loss,
        'generation_loss': args.generation_loss,
        'testing':args.testing
    }

    pp.pprint(config)


    return config
