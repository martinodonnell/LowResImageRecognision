import torchvision
import sys
sys.path.append("..")
from models.mtlearningNonClassic import MTLNC_Shared_FC_B,MTLNC_Seperate_FC,MTLNC_Shared_FC_A,MTLNC_Shared_FC_C
from unitTests.models.testUtil import check_layers_multitask


def test_MTLNC_Shared_FC_B_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLNC_Shared_FC_B(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)


def test_MTLNC_Seperate_FC_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLNC_Seperate_FC(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)


def test_MTLNC_Shared_FC_A_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLNC_Shared_FC_A(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)

def test_MTLNC_Shared_FC_C_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLNC_Shared_FC_C(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)