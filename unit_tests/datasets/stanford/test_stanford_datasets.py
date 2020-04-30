import torchvision
import sys
import os
if ('..' not in sys.path) : sys.path.append("..")
from config import STANFORD_CARS_TRAIN,STANFORD_CARS_TEST,STANFORD_CARS_TRAIN_ANNOS,STANFORD_CARS_TEST_ANNOS
from datasets.stanford.stanford_datasets import StanfordCarsDatasetV1,StanfordCarsDatasetV2,StanfordCarsDatasetV3
from torchvision import transforms
import torch

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    if os.getcwd().split('/')[-1] != 'LowResImageRecognision' : os.chdir("..")
    print('after',os.getcwd())

def test_check_stanford_v1_dataset_split_low_version():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )


    train_dataset = StanfordCarsDatasetV1(STANFORD_CARS_TRAIN, STANFORD_CARS_TRAIN_ANNOS, basic_transform, (224,224),True)
    test_dataset = StanfordCarsDatasetV1(STANFORD_CARS_TEST, STANFORD_CARS_TEST_ANNOS, basic_transform, (224,224),True)

    assert len(train_dataset) ==8144
    assert len(test_dataset) ==8041

def test_check_stanford_v1_dataset_split_high_version():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )


    train_dataset = StanfordCarsDatasetV1(STANFORD_CARS_TRAIN, STANFORD_CARS_TRAIN_ANNOS, basic_transform, (224,224),False)
    test_dataset = StanfordCarsDatasetV1(STANFORD_CARS_TEST, STANFORD_CARS_TEST_ANNOS, basic_transform, (224,224),False)

    assert len(train_dataset) ==8144
    assert len(test_dataset) ==8041

def test_stanford_v1_data():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = StanfordCarsDatasetV1(STANFORD_CARS_TRAIN, STANFORD_CARS_TRAIN_ANNOS, basic_transform, (224,224),False)
    
    #Get first output
    sample = next(iter(train_dataset))

    #Check dimensions of image
    assert (sample[0].size()[0] == 3)
    assert (sample[0].size()[1] == 224)
    assert (sample[0].size()[2] == 224)

    #Check datatype of target(int)
    assert type(sample[1])== int



def test_check_stanford_v2_dataset_split_high_version():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )


    train_dataset = StanfordCarsDatasetV2(STANFORD_CARS_TRAIN, STANFORD_CARS_TRAIN_ANNOS, basic_transform, (224,224),False)
    test_dataset = StanfordCarsDatasetV2(STANFORD_CARS_TEST, STANFORD_CARS_TEST_ANNOS, basic_transform, (224,224),False)

    assert len(train_dataset) ==8144
    assert len(test_dataset) ==8041

def test_check_stanford_v2_dataset_split_low_version():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )


    train_dataset = StanfordCarsDatasetV2(STANFORD_CARS_TRAIN, STANFORD_CARS_TRAIN_ANNOS, basic_transform, (224,224),True)
    test_dataset = StanfordCarsDatasetV2(STANFORD_CARS_TEST, STANFORD_CARS_TEST_ANNOS, basic_transform, (224,224),True)

    assert len(train_dataset) ==8144
    assert len(test_dataset) ==8041

def test_stanford_v2_data():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = StanfordCarsDatasetV2(STANFORD_CARS_TRAIN, STANFORD_CARS_TRAIN_ANNOS, basic_transform, (224,224),False)
    
    #Get first output
    sample = next(iter(train_dataset))

    #Check dimensions of image
    assert (sample[0].size()[0] == 3)
    assert (sample[0].size()[1] == 224)
    assert (sample[0].size()[2] == 224)

    #Check datatype of target(int)
    assert type(sample[1])== int
    assert type(sample[2])== int


def test_check_stanford_v3_dataset_split():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )


    train_dataset = StanfordCarsDatasetV3(STANFORD_CARS_TRAIN, STANFORD_CARS_TRAIN_ANNOS, basic_transform, (224,224),False)
    test_dataset = StanfordCarsDatasetV3(STANFORD_CARS_TEST, STANFORD_CARS_TEST_ANNOS, basic_transform, (224,224),False)

    assert len(train_dataset) ==8144*2
    assert len(test_dataset) ==8041*2

def test_stanford_v3_data():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = StanfordCarsDatasetV3(STANFORD_CARS_TRAIN, STANFORD_CARS_TRAIN_ANNOS, basic_transform, (224,224),False)
    
    #Get first output
    sample = next(iter(train_dataset))

    #Check dimensions of image
    assert (sample[0].size()[0] == 3)
    assert (sample[0].size()[1] == 224)
    assert (sample[0].size()[2] == 224)

    #Check datatype of target(int)
    assert type(sample[1])== int








