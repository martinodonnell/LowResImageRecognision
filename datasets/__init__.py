from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from datasets.stanford.StanfordDataset import StanfordCarsDatasetV1,StanfordCarsDatasetV2,StanfordCarsDatasetV3
from datasets.boxcars.boxcars_datasets import BoxCarsDatasetV1,BoxCarsDatasetV1_2,BoxCarsDatasetV2,BoxCarsDatasetV3

from config import BOXCARS_IMAGES_IMAGES,STANFORD_CARS_TRAIN,STANFORD_CARS_TEST,STANFORD_CARS_TRAIN_ANNOS,STANFORD_CARS_TEST_ANNOS
from exceptions.exceptions import InvalidDatasetVersion


fine_grain_model_ids = [2,9,10,11,13,14]
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

    train_dataset,test_dataset = get_train_test_dataset(config,train_transform,test_transform)

    config = add_class_numbers_to_config(config) 

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

    _,test_dataset = get_train_test_dataset(config,test_transform,test_transform,'test')      

    config = add_class_numbers_to_config(config)
    
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


def add_class_numbers_to_config(config):
    if config['dataset_version'] in [1,5]:
        config['num_classes']=196
        config['num_makes']=49
        config['num_models']=18
        config['num_submodels']=16
        config['num_generations']=1 
    else:
        config['num_classes']=107
        config['num_makes']=16
        config['num_models']=68
        config['num_submodels']=6
        config['num_generations']=7

    return config


def get_train_test_dataset(config,train_transform,test_transform,part="validation"):
    if(config['dataset_version']==1):#Stanford Cars Dataset
        train_imgdir = STANFORD_CARS_TRAIN
        test_imgdir = STANFORD_CARS_TEST
        train_annopath = STANFORD_CARS_TRAIN_ANNOS
        test_annopath = STANFORD_CARS_TEST_ANNOS
        
        if config['train_test_version'] in [1]:  
            print("Dataset: Stanford Normal")       
            train_dataset = StanfordCarsDatasetV1
            test_dataset = StanfordCarsDatasetV1
        else:
            print("Dataset: Stanford MTL")      

            train_dataset = StanfordCarsDatasetV2
            test_dataset = StanfordCarsDatasetV2  
        
        train_dataset = train_dataset(train_imgdir, train_annopath, train_transform, config['imgsize'],config['ds-stanford'])
        test_dataset = test_dataset(test_imgdir, test_annopath, test_transform, config['imgsize'],config['ds-stanford'])


    elif(config['dataset_version']==2):#BoxCars Dataset
        imgdir =  BOXCARS_IMAGES_IMAGES

        if config['train_test_version'] in [1,7]:    
            print("Dataset: Boxcars Normal")       
     
            train_dataset = BoxCarsDatasetV1
            test_dataset = BoxCarsDatasetV1
        else:
            print("Dataset: Boxcars MTL")       

            train_dataset = BoxCarsDatasetV2
            test_dataset = BoxCarsDatasetV2
        
        train_dataset = train_dataset(imgdir, train_transform, config['imgsize'],config['boxcar_split'],'train')
        test_dataset = test_dataset(imgdir, test_transform, config['imgsize'],config['boxcar_split'],part)
    
    elif(config['dataset_version']==3):#BoxCars Dataset with augmentation
        print("Dataset: Boxcars Augmentation")  
        imgdir =  BOXCARS_IMAGES_IMAGES
        train_dataset = BoxCarsDatasetV3(imgdir, train_transform, config['imgsize'],config['boxcar_split'],'train')
        test_dataset = BoxCarsDatasetV3(imgdir, test_transform, config['imgsize'],config['boxcar_split'],part)

    elif(config['dataset_version']==4):#Train with a lessor amoutn of training
        print("Dataset: Boxcars Sample Limiter")  
        imgdir =  BOXCARS_IMAGES_IMAGES
        train_dataset = BoxCarsDatasetV1_2(imgdir, train_transform, config['imgsize'],config['boxcar_split'],'train',config['train_samples'],config['train_samples_percent'])
        test_dataset = BoxCarsDatasetV1_2(imgdir, test_transform, config['imgsize'],config['boxcar_split'],part,config['train_samples'],config['train_samples_percent'])
    elif(config['dataset_version']==5):#Mixed Stanford Dataset
        train_imgdir = STANFORD_CARS_TRAIN
        test_imgdir = STANFORD_CARS_TEST
        train_annopath = STANFORD_CARS_TRAIN_ANNOS
        test_annopath = STANFORD_CARS_TEST_ANNOS

        train_dataset = StanfordCarsDatasetV3(train_imgdir, train_annopath, train_transform, config['imgsize'],config['ds-stanford'])
        test_dataset = StanfordCarsDatasetV3(test_imgdir, test_annopath, test_transform, config['imgsize'],config['ds-stanford'])
    else:
        raise InvalidDatasetVersion("Invalid dataset version") 

    return train_dataset,test_dataset