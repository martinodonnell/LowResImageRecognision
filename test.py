import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time

from datasets import prepare_test_loader
from models import construct_model
from config import SAVE_FOLDER,CONFUSION_MATRIX 
from trainTest import get_train_test_methods
from trainTestUtil import get_args,load_weight,get_loss_function


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model_path = os.path.join(SAVE_FOLDER, str(config['model_id']).zfill(3) + '_model.pth')

    test_loader, confusion_matrix = prepare_test_loader(config)

    model = construct_model(config, config['num_classes'], config['num_makes'], config['num_models'],
                            config['num_submodels'], config['num_generations'])
    load_weight(model, model_path, device)
    model = model.to(device)

    _, test_fn = get_train_test_methods(config)
    
    #Get loss function
    loss_function = get_loss_function(config)

    valres = test_fn(model, test_loader, device, config, confusion_matrix,loss_function)

    #Write confusion matrix to output  with each on in different sheet
    writer = pd.ExcelWriter(os.path.join(CONFUSION_MATRIX, str(config['model_id']) + "_CM_"+time.strftime("%Y%m%d-%H%M%S")+".xlsx") , engine='xlsxwriter')
    for key,values in confusion_matrix.items():
        df_cm = pd.DataFrame(values.numpy(), range(values.shape[0]), range(values.shape[1]))
        df_cm.to_excel(writer, sheet_name=key)
    writer.save()


if __name__ == '__main__':
    config = get_args()
    main(config)
