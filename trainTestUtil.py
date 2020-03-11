from trainTest.testMethods import test_v1, test_v2, test_v3, test_v4, test_v6, test_v7, test_v8, test_v9
from trainTest.trainMethods import train_v1, train_v2, train_v3, train_v4, train_v6, train_v7, train_v8, train_v9
import os
from config import SAVE_FOLDER
import pandas as pd

from trainTest.TrainTest5 import test_v5, train_v5


def get_train_test_methods(config):
    if config['model_version'] in [2]:  # Multitask learning (2 features) Boxcars
        print("Train/Test Version 2 for boxcars (Multitask learning - 2 features) ")
        return train_v2, test_v2

    elif config['model_version'] in [9, 10, 13, 14]:  # Multitask learning (3 features) Boxcars
        print("Train/Test Version 4 for boxcars (Multitask learning - 3 features)")
        return train_v4, test_v4

    elif config['model_version'] in [11]:  # Multitask learning with classic ml format

        # Get model which will seperatly get prediction for each fine-grain attribute
        if config['train_test_version'] == 1:
            print("Train/Test Version 5_1 for boxcars (Multitask learning - 3 features classic)")
            return train_v5, test_v5

        # Get prediction for just make
        elif config['train_test_version'] == 2:
            print("Train/Test Version 5_2 for boxcars (Multitask learning - 3 features classic)")
            return train_v6, test_v6

            # Get prediction for just model
        elif config['train_test_version'] == 3:
            print("Train/Test Version 5_3 for boxcars (Multitask learning - 3 features classic)")
            return train_v7, test_v7

            # Get prediction for just submodel
        elif config['train_test_version'] == 4:
            print("Train/Test Version 5_4 for boxcars (Multitask learning - 3 features classic)")
            return train_v8, test_v8

            # Get prediction for just generation
        elif config['train_test_version'] == 5:
            print("Train/Test Version 5_5 for boxcars (Multitask learning - 3 features classic)")
            return train_v9, test_v9

    elif config['model_version'] in [8]:  # Multitask learning 2 features Stanford
        print("Train/Test Version 3 for stanford (Multitask learning)")
        return train_v3, test_v3
    else:  # normal
        print("Train/Test Version 1 for normal models")
        return train_v1, test_v1


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
    while os.path.isfile(csv_history_filepath):
        config['model_id'] = config['model_id'] + 1
        csv_history_filepath, model_best_filepath = get_output_filepaths(config['model_id'])
    print("Current ID:", config['model_id'])

    # Set up blank csv in save folder
    if config['model_version'] in [2,8]:
        # Multitask learning (2 features) Boxcars or #Multitask learning 2 features Stanford
        df = pd.DataFrame(columns=['train_loss', 'train_acc', 'train_make_loss', 'train_make_acc', 'train_model_loss',
                                   'train_model_acc', 'train_time', 'val_loss', 'val_acc', 'val_make_acc',
                                   'val_make_loss', 'val_model_acc', 'val_model_loss', 'val_time', 'lr', 'overwritten',
                                   'epoch'])
    elif config['model_version'] in [9, 10, 13, 14]:  # Multitask learning (3 features) Boxcars
        df = pd.DataFrame(columns=['train_loss', 'train_acc', 'train_make_loss', 'train_make_acc', 'train_model_loss',
                                   'train_model_acc', 'train_submodel_loss', 'train_submodel_acc', 'train_time',
                                   'val_loss', 'val_acc', 'val_make_acc', 'val_make_loss', 'val_model_acc',
                                   'val_model_loss', 'val_submodel_acc', 'val_submodel_loss', 'val_time', 'lr',
                                   'overwritten', 'epoch'])
    elif config['model_version'] in [11]:  # Multitask learning with classifc ml format
        df = pd.DataFrame(columns=['train_loss', 'train_acc', 'train_make_loss', 'train_make_acc', 'train_model_loss',
                                   'train_model_acc', 'train_submodel_loss', 'train_submodel_acc',
                                   'train_generation_loss', 'train_generation_acc', 'train_time', 'val_loss', 'val_acc',
                                   'val_make_acc', 'val_make_loss', 'val_model_acc', 'val_model_loss',
                                   'val_submodel_acc', 'val_submodel_loss', 'val_generation_acc', 'val_generation_loss',
                                   'val_time', 'lr', 'overwritten', 'epoch'])
    else:  # Normal
        df = pd.DataFrame(
            columns=['train_loss', 'train_acc', 'train_time', 'val_loss', 'val_acc', 'val_time', 'lr', 'overwritten',
                     'epoch'])
    df.to_csv(csv_history_filepath)

    return config, csv_history_filepath, model_best_filepath
