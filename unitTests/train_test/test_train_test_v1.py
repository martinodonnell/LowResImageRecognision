import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from train_test.train_test_1 import train_v1,test_v1 as t_test_v1
from models.generic_networks import BoxCarsBase
from unitTests.train_test.utils import compare_model_keys_same,compare_model_keys_diff,TestingRandomDatasetSingle
from datasets import gen_confusion_matrixes

def test_train_v1():

    #Create two of the same models to see if weighst are updated in each layer during training
    num_classes = 196
    base = torchvision.models.vgg16(pretrained=True, progress=True)         
    model1 = BoxCarsBase(base,num_classes)
    model2 = BoxCarsBase(base,num_classes)
    assert compare_model_keys_same(model1,model2)

    ep = 1
    optimizer = optim.Adam(model1.parameters())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    loss_function = F.cross_entropy


    train_loader = DataLoader(TestingRandomDatasetSingle(1),batch_size=1)   

    #use train method to update weights in model and get results
    results = train_v1(ep, model1, optimizer, train_loader, device, None,loss_function)


    #Checking the number of results are all equal to zero
    assert len(results)==3
    for value in results.values():
        assert value is not 0

    #Checking that model has been updated
    assert compare_model_keys_diff(model1,model2)


def test_test_v1():

    #Create two of the same models to see if weighst are updated in each layer during training
    num_classes = 196
    base = torchvision.models.vgg16(pretrained=True, progress=True)         
    model1 = BoxCarsBase(base,num_classes)

    ep = 1
    optimizer = optim.Adam(model1.parameters())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    loss_function = F.cross_entropy

    config = {
        'main_loss':1,
        'make_loss':1,
        'model_loss':1,
        'submodel_loss':1,
        'generation_loss':1,
        'num_classes':num_classes,
        'num_makes':num_makes,
        'num_models':num_models,
        'num_submodels':num_submodels,
        'num_generations':num_generations 
    }

    confusion_matrix = gen_confusion_matrixes(config)
    train_loader = DataLoader(TestingRandomDatasetSingle(1),batch_size=1)   

    #use train method to update weights in model and get results
    results = t_test_v1(ep, model1, optimizer, train_loader, device, None,loss_function)


    #Checking the number of results are all equal to zero
    assert len(results)==3
    

