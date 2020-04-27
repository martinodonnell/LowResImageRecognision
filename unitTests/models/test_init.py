import sys
sys.path.append("..")
import pytest
import torchvision
from models.generic_networks import BoxCarsBase,StanfordBase,AlexnetBase
from models.multitask_networks import MTLC_Shared_FC,MTLC_Seperate_FC,Old_MTLC_Seperate_FC,MTLC_Shared_FC_B,MTLC_Shared_FC_C
from models.old_auxiliary_networks import MTLNC_Shared_FC_B,MTLNC_Seperate_FC,MTLNC_Shared_FC_A,MTLNC_Shared_FC_C
from models.pooling_networks import ChannelPoolingNetwork,SpatiallyWeightedPoolingNetwork
from models.auxillary_learning import AuxillaryLearning_A,AuxillaryLearning_B,AuxillaryLearning_C
from exceptions.exceptions import InvalidModelVersion
from models import construct_model



def compare_model_keys(model1,model2):
    for key1,key2 in zip(model1.state_dict(),model2.state_dict()):
        if key1!=key2:
            return False
    return True

def test_construct_model_stanford():

    num_class = 196
    num_make = 1
    num_model = 1
    num_submodel = 1
    num_generations = 1
    base = torchvision.models.vgg16(pretrained=True, progress=True) 

    model1 = StanfordBase(base, num_class)
    model2 = construct_model(7, num_class,num_make,num_model,num_submodel,num_generations)

    assert compare_model_keys(model1,model2)


def test_construct_model_boxcars():

    num_class = 196
    num_make = 1
    num_model = 1
    num_submodel = 1
    num_generations = 1
    base = torchvision.models.vgg16(pretrained=True, progress=True) 

    model1 = BoxCarsBase(base, num_class)
    model2 = construct_model(1, num_class,num_make,num_model,num_submodel,num_generations)

    assert compare_model_keys(model1,model2)

def test_construct_model_invalid_version():

    num_class = 196
    num_make = 1
    num_model = 1
    num_submodel = 1
    num_generations = 1
    with pytest.raises(InvalidModelVersion):
        construct_model(-1, num_class,num_make,num_model,num_submodel,num_generations)





