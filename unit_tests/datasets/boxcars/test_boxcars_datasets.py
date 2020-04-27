import torchvision
import sys
import os
sys.path.append("..")
from config import BOXCARS_IMAGES_IMAGES
from datasets.boxcars.boxcars_datasets import BoxCarsDatasetV1,BoxCarsDatasetV1_2,BoxCarsDatasetV2,BoxCarsDatasetV3
from torchvision import transforms
import torch

os.chdir("..")

def test_check_boxcars_v1_dataset_split():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = BoxCarsDatasetV1(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train')
    val_dataset = BoxCarsDatasetV1(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','validation')
    test_dataset = BoxCarsDatasetV1(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','test')

    assert len(train_dataset) ==51691
    assert len(val_dataset) ==2763
    assert len(test_dataset) ==39149

def test_boxcars_v1_data():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = BoxCarsDatasetV1(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train')
    
    #Get first output
    sample = next(iter(train_dataset))

    #Check dimensions of image
    assert (sample[0].size()[0] == 3)
    assert (sample[0].size()[1] == 224)
    assert (sample[0].size()[2] == 224)

    #Check datatype of target(int)
    assert type(sample[1])== int


def test_check_boxcars_v1_2_dataset_split():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train',1,True)
    val_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','validation',1,False)
    test_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','test',1,False)

    assert len(train_dataset) ==51691
    assert len(val_dataset) ==2763
    assert len(test_dataset) ==39149

def test_boxcars_v1_2_decrease_train_dinstinct():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    total_samples = 107
    num_samples_decrease = 1
    train_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train',num_samples_decrease,False)
    assert len(train_dataset) ==total_samples*num_samples_decrease

    num_samples_decrease = 10
    train_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train',num_samples_decrease,False)
    assert len(train_dataset) ==total_samples*num_samples_decrease

    num_samples_decrease = 32
    train_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train',num_samples_decrease,False)
    assert len(train_dataset) ==total_samples*num_samples_decrease

    num_samples_decrease = 56
    train_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train',num_samples_decrease,False)
    assert len(train_dataset) ==total_samples*num_samples_decrease

    #Some classes don't have more than 60 classes so number will not match up 100%
    num_samples_decrease = 80
    train_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train',num_samples_decrease,False)
    assert len(train_dataset) <=total_samples*num_samples_decrease

def test_boxcars_v1_2_decrease_train_percent():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )

    total_samples = 51691
    num_samples_decrease = 1
    train_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train',num_samples_decrease,True)
    assert len(train_dataset) <=total_samples*num_samples_decrease

    num_samples_decrease = 0.5
    train_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train',num_samples_decrease,True)
    assert len(train_dataset) <=total_samples*num_samples_decrease

    num_samples_decrease = 0.25
    train_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train',num_samples_decrease,True)
    assert len(train_dataset) <=total_samples*num_samples_decrease
    

def test_boxcars_v1_2_data():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = BoxCarsDatasetV1_2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train',1,False)
    
    #Get first output
    sample = next(iter(train_dataset))

    #Check dimensions of image
    assert (sample[0].size()[0] == 3)
    assert (sample[0].size()[1] == 224)
    assert (sample[0].size()[2] == 224)

    #Check datatype of target(int)
    assert type(sample[1])== int


def test_check_boxcars_v2_dataset_split():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = BoxCarsDatasetV2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train')
    val_dataset = BoxCarsDatasetV2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','validation')
    test_dataset = BoxCarsDatasetV2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','test')

    assert len(train_dataset) ==51691
    assert len(val_dataset) ==2763
    assert len(test_dataset) ==39149

def test_boxcars_v2_data():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = BoxCarsDatasetV2(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train')
    
    #Get first output
    sample = next(iter(train_dataset))

    #Check dimensions of image
    assert (sample[0].size()[0] == 3)
    assert (sample[0].size()[1] == 224)
    assert (sample[0].size()[2] == 224)

    #Check datatype of target(int)
    assert type(sample[1])== int
    assert type(sample[2])== int
    assert type(sample[3])== int
    assert type(sample[4])== int
    assert type(sample[5])== int

def test_check_boxcars_v3_dataset_split():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = BoxCarsDatasetV3(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train')
    val_dataset = BoxCarsDatasetV3(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','validation')
    test_dataset = BoxCarsDatasetV3(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','test')

    assert len(train_dataset) ==51691
    assert len(val_dataset) ==2763
    assert len(test_dataset) ==39149

def test_boxcars_v3_data():

    basic_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
        ]
    )
    train_dataset = BoxCarsDatasetV3(BOXCARS_IMAGES_IMAGES, basic_transform, (224,224),'hard','train')
    
    #Get first output
    sample = next(iter(train_dataset))

    #Check dimensions of image
    assert (sample[0].size()[0] == 3)
    assert (sample[0].size()[1] == 224)
    assert (sample[0].size()[2] == 224)

    #Check datatype of target(int)
    assert type(sample[1])== int








