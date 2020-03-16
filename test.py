import argparse
import json
import os
import pprint as pp
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from datasets import prepare_test_loader
from models import construct_model
from config import SAVE_FOLDER

from trainTestUtil import get_train_test_methods


# -------------------------------
# Classic multitask learning END
# -------------------------------


def load_weight(model, path, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    config = {
        'model_id': args.model_id,
        'model_version': args.model_version,
        'dataset_version': args.dataset_version,
        'train_test_version': args.train_test_version,
        'train_samples':args.train_samples,

        'imgsize': (244, 244),
        'boxcar_split': 'hard',
        'test_batch_size': 60,
        
        'main_loss': args.main_loss,
        'make_loss': args.make_loss,
        'model_loss': args.model_loss,
        'submodel_loss': args.submodel_loss,
        'generation_loss': args.generation_loss,
    }

    pp.pprint(config)

    model_path = os.path.join(SAVE_FOLDER, str(config['model_id']).zfill(3) + '_model.pth')

    test_loader, confusion_matrix = prepare_test_loader(config)

    model = construct_model(config, config['num_classes'], config['num_makes'], config['num_models'],
                            config['num_submodels'], config['num_generations'])
    load_weight(model, model_path, device)
    model = model.to(device)

    _, test_fn = get_train_test_methods(config)

    valres = test_fn(model, test_loader, device, config, confusion_matrix)

    # print confusion matrixes to kelvin log files
    for key, value in confusion_matrix.items():
        print('-----', key, '-----')
        matrix_data = '['
        for x in value:
            matrix_data += '['
            for y in x:
                matrix_data += str(y.item()) + ', '
            matrix_data = matrix_data[:-2] + '],\n'
        matrix_data = matrix_data[:-2] + ']\n'
        print(matrix_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script for Cars dataset')

    parser.add_argument('--model-id', default=15, type=int, required=True,
                        help='id to lined to previous model to fine tune. Required if it is a fine tune task')
    parser.add_argument('--model-version', type=int, required=True,
                        help='Classification version \n')
    parser.add_argument('--train-test-version', default=1, type=int,
                        help='Some models have more than one test_train setup giving different training and test '
                             'abilities)\n')
    parser.add_argument('--dataset-version', default=1, type=int, choices=[1, 2, 3,4], required=True,
                        help='Classification version (default: 1)\n'
                             '1. Stanford Dataset\n'
                             '2. BoxCar Dataset')
                             

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

    # Not used but added so I didn't need to add conditions in bash file to remove them which would take too long and
    # wasting time
    parser.add_argument('--epochs', default=150, type=int,
                        help='training epochs (default: 60)')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='training batch size (default: 32)')
    parser.add_argument('--train-samples', default=1,
                        help='required if set dataset-version to 4 (default: 1). Determines how many training samples to train the model with during testing from boxcars')
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
    parser.add_argument('--finetune', default=False, action='store_true',
                        help='fine tune an existing model (default: False)')
    parser.add_argument('--finetune-stan-box', default=False, action='store_true',
                        help='Fine tune stanfor dmodel with boxcars default: False)')                        
    parser.add_argument('--fine-tune-id', type=int,
                        help='id to lined to previous model to fine tune. Required if it is a fine tune task')
                        
    args = parser.parse_args()

    main(args)
