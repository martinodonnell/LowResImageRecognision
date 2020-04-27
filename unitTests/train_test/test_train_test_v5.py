import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from train_test.train_test_5 import train_v5, test_v5 as t_test_v5
from models.multitask_networks import MTLC_Shared_FC
from unitTests.train_test.utils import compare_model_keys_same,compare_model_keys_diff,TestingRandomDatasetMulti
from datasets import gen_confusion_matrixes

def test_test_v5():

    #Create two of the same models to see if weighst are updated in each layer during training
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2

    base = torchvision.models.vgg16(pretrained=True, progress=True)         
    model1 = MTLC_Shared_FC(base,num_classes, num_makes, num_models,num_submodels,num_generations)

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


    test_loader = DataLoader(TestingRandomDatasetMulti(1),batch_size=1)   
    confusion_matrix = gen_confusion_matrixes(config)

    results = t_test_v5(model1, test_loader, device, config,confusion_matrix,loss_function)


    #Checking the number of results are all equal to zero
    assert len(results)==11


def test_train_v5():

    #Create two of the same models to see if weighst are updated in each layer during training
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2

    base = torchvision.models.vgg16(pretrained=True, progress=True)         
    model1 = MTLC_Shared_FC(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    model2 = MTLC_Shared_FC(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    assert compare_model_keys_same(model1,model2)

    ep = 1
    optimizer = optim.Adam(model1.parameters())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    loss_function = F.cross_entropy
    config = {
        'make_loss':1,
        'model_loss':1,
        'submodel_loss':1,
        'generation_loss':1
    }


    train_loader = DataLoader(TestingRandomDatasetMulti(1),batch_size=1)   

    #use train method to update weights in model and get results
    results = train_v5(ep, model1, optimizer, train_loader, device, config,loss_function)


    #Checking the number of results are all equal to zero
    assert len(results)==11
    for value in results.values():
        assert value is not 0

    #Checking that model has been updated
    assert compare_model_keys_diff(model1,model2)
    

