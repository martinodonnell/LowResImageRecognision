import os
import scipy.io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import numpy as np
import torch

from datasets.StanfordDataset import CarsDatasetV1,CarsDatasetV2
from datasets.BoxCarsDataset import BoxCarsDatasetV1,BoxCarsDatasetV2
from config import BOXCARS_DATASET_ROOT,BOXCARS_IMAGES_IMAGES,BOXCARS_CLASSIFICATION_SPLITS,BOXCARS_DATASET,BOXCARS_HARD_CLASS_NAMES
from config import STANFORD_CARS_TRAIN,STANFORD_CARS_TEST,STANFORD_CARS_TRAIN_ANNOS,STANFORD_CARS_TEST_ANNOS,STANFORD_CARS_CARS_META


fine_grain_model_ids = [2,9,10,11]
def prepare_loader(config):

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    if(config['dataset_version']==1):#Stanford Cars Dataset
        train_imgdir = STANFORD_CARS_TRAIN
        test_imgdir = STANFORD_CARS_TEST

        train_annopath = STANFORD_CARS_TRAIN_ANNOS
        test_annopath = STANFORD_CARS_TEST_ANNOS
        
        if(config['model_version']!=8):         
            train_dataset = CarsDatasetV1(train_imgdir, train_annopath, train_transform, config['imgsize'])
            test_dataset = CarsDatasetV1(test_imgdir, test_annopath, test_transform, config['imgsize'])
        else:
            train_dataset = CarsDatasetV2(train_imgdir, train_annopath, train_transform, config['imgsize'])
            test_dataset = CarsDatasetV2(test_imgdir, test_annopath, test_transform, config['imgsize'])
            
        
        config['num_classes']=196
        config['num_makes']=49
        config['num_models']=18
        config['num_submodels']=1
        config['num_generations']=1
        


    elif(config['dataset_version']==2):#BoxCars Dataset
        imgdir = test_imgdir =  BOXCARS_IMAGES_IMAGES

        if(config['model_version'] in fine_grain_model_ids):
            train_dataset = BoxCarsDatasetV2(imgdir, train_transform, config['imgsize'],config['boxcar_split'],'train')
            test_dataset = BoxCarsDatasetV2(imgdir, test_transform, config['imgsize'],config['boxcar_split'],'validation')
        else:
            train_dataset = BoxCarsDatasetV1(imgdir, train_transform, config['imgsize'],config['boxcar_split'],'train')
            test_dataset = BoxCarsDatasetV1(imgdir, test_transform, config['imgsize'],config['boxcar_split'],'validation')
    
        config['num_classes']=107
        config['num_makes']=16
        config['num_models']=68
        config['num_submodels']=6
        config['num_generations']=7
    else:
        print("No dataset. Leaving")
        exit(1)   

    confusion_matrixes = gen_confusion_matrixes(config)

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              pin_memory=False,
                              num_workers=12)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['test_batch_size'],
                             shuffle=False,
                             pin_memory=False,
                             num_workers=12)

    return train_loader, test_loader,confusion_matrixes


def prepare_test_loader(config):

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    if(config['dataset_version']==1):#Stanford Cars Dataset
        test_imgdir = STANFORD_CARS_TEST
        test_annopath = STANFORD_CARS_TEST_ANNOS
        
        if(config['model_version']!=8): #Fine grain dataset
            test_dataset = CarsDatasetV1(test_imgdir, test_annopath, test_transform, config['imgsize'])
        else:#Full labels
            test_dataset = CarsDatasetV2(test_imgdir, test_annopath, test_transform, config['imgsize'])
            
        config['num_classes']=196
        config['num_makes']=49
        config['num_models']=18
        config['num_submodels']=1
        config['num_generations']=1


    elif(config['dataset_version']==2):#BoxCars Dataset
        imgdir = test_imgdir =  BOXCARS_IMAGES_IMAGES

        if(config['model_version']in fine_grain_model_ids):#Fine grain dataset
            test_dataset = BoxCarsDatasetV2(imgdir, test_transform, config['imgsize'],config['boxcar_split'],'test')
        else:#Full labels
            test_dataset = BoxCarsDatasetV1(imgdir, test_transform, config['imgsize'],config['boxcar_split'],'test')
       
        config['num_classes']=107 
        config['num_makes']=16
        config['num_models']=68
        config['num_submodels']=6
        config['num_generations']=7
    else:
        print("No dataset. Leaving")
        exit(1)   

    
    confusion_matrixes = gen_confusion_matrixes(config)

    test_loader = DataLoader(test_dataset,
                             batch_size=config['test_batch_size'],
                             shuffle=False,
                             pin_memory=False,
                             num_workers=12)

    return test_loader,confusion_matrixes


def gen_confusion_matrixes(config):
    confusion_matrix = {}
    confusion_matrix['total'] = torch.zeros(config['num_classes'], config['num_classes'])
    confusion_matrix['make'] = torch.zeros(config['num_makes'], config['num_makes'])
    confusion_matrix['model'] = torch.zeros(config['num_models'], config['num_models'])
    confusion_matrix['submodel'] = torch.zeros(config['num_submodels'], config['num_submodels'])
    confusion_matrix['generation'] = torch.zeros(config['num_generations'], config['num_generations'])

    return confusion_matrix

