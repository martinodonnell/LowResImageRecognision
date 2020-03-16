from trainTest.testMethods import test_v1, test_v2, test_v3, test_v4,test_v5
from trainTest.trainMethods import train_v1, train_v2, train_v3, train_v4,train_v5
from trainTest.TrainTest6 import train_v6,test_v6
import os
from config import SAVE_FOLDER
import pandas as pd

def get_train_test_methods(config):

    #Normal model
    if config['train_test_version'] == 1:
        print("Train/Test Version 1 for normal models")
        return train_v1, test_v1

    #---- Multitask learning --- 
    #Not used. Use model 6 and change multiplications on loss
    elif config['train_test_version'] == 2:
        print("Train/Test Version 2 for BOXCARS (NON Class Multitask learning - 2 features) ")
        return train_v2, test_v2
    elif config['train_test_version'] == 3: 
        print("Train/Test Version 3 for STANFORD (Multitask learning)")
        return train_v3, test_v3
    # Not used. Use model 6 and change multiplications on loss
    elif config['train_test_version'] == 4:
        print("Train/Test Version 4 for BOXCARS (Multitask learning - 3 features)")
        return train_v4, test_v4
    elif config['train_test_version'] == 5:
        print("Train/Test Version 5 for BOXCARS (Multitask learning - 4 features classic)")
        return train_v5, test_v5
    elif config['train_test_version'] == 6:
        print("Train/Test Version 6 for BOXCARS (Multitask learning - 4 features Non classic)")
        return train_v6, test_v6
    else:
        print(config['train_test_version'], "is not a valid trainTest method(get_train_test_methods)")
        exit(1) 

def get_output_filepaths(id):
    id_string = str(id).zfill(3)
    csv_history_filepath = os.path.join(SAVE_FOLDER, id_string + '_history.csv')
    model_best_filepath = os.path.join(SAVE_FOLDER, id_string + '_model.pth')

    return csv_history_filepath, model_best_filepath


def set_up_output_filepaths(config):
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
    if config['train_test_version'] in [1]:
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
