import os
import scipy.io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import numpy as np

from datasets.StanfordDataset import CarsDatasetV1,CarsDatasetV2
from datasets.BoxCarsDataset import BoxCarsDatasetV1,BoxCarsDatasetV2
from config import BOXCARS_DATASET_ROOT,BOXCARS_IMAGES_IMAGES,BOXCARS_CLASSIFICATION_SPLITS,BOXCARS_DATASET,BOXCARS_HARD_CLASS_NAMES
from config import STANFORD_CARS_TRAIN,STANFORD_CARS_TEST,STANFORD_CARS_TRAIN_ANNOS,STANFORD_CARS_TEST_ANNOS,STANFORD_CARS_CARS_META

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
            
        multi_nums = {
            'num_classes':196, 
            'num_makes':49,
            'num_models':18,
            'num_submodels':1,
            'generation':1,
        }      


    elif(config['dataset_version']==2):#BoxCars Dataset
        imgdir = test_imgdir =  BOXCARS_IMAGES_IMAGES

        if(config['model_version']==2):
            train_dataset = BoxCarsDatasetV2(imgdir, train_transform, config['imgsize'],config['boxcar_split'],'train')
            test_dataset = BoxCarsDatasetV2(imgdir, test_transform, config['imgsize'],config['boxcar_split'],'validation')
        else:
            train_dataset = BoxCarsDatasetV1(imgdir, train_transform, config['imgsize'],config['boxcar_split'],'train')
            test_dataset = BoxCarsDatasetV1(imgdir, test_transform, config['imgsize'],config['boxcar_split'],'validation')
       
        multi_nums = {
            'num_classes':107, 
            'num_makes':16,
            'num_models':68,
            'num_submodels':6,
            'generation':7,
        }
    else:
        print("No dataset. Leaving")
        exit(1)   

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

    return multi_nums,train_loader, test_loader


def prepare_test_loader(config):

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    if(config['dataset_version']==1):
        print("NOT SET UP YET")
        exit(1)
        train_imgdir = STANFORD_CARS_TRAIN
        test_imgdir = STANFORD_CARS_TEST

        train_annopath = STANFORD_CARS_TRAIN_ANNOS
        test_annopath = STANFORD_CARS_TEST_ANNOS

        test_dataset = CarsDataset(test_imgdir, test_annopath, test_transform, config['imgsize'],config['dataset_version'],config['boxcar_split'],'test')

    elif(config['dataset_version']==2):
        imgdir = test_imgdir =  BOXCARS_IMAGES_IMAGES
        test_dataset = BoxCarsDatasetV1(imgdir, test_transform, config['imgsize'],config['boxcar_split'],'test')
    else:
        print("No dataset. Leaving")
        exit(1) 

    test_loader = DataLoader(test_dataset,
                             batch_size=config['test_batch_size'],
                             shuffle=False,
                             pin_memory=False,
                             num_workers=12)

    return test_loader
    

