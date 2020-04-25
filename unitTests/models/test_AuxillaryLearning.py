import torchvision
import sys
sys.path.append("..")
from models.auxillary_learning import AuxillaryLearning_A,AuxillaryLearning_B,AuxillaryLearning_C
from unitTests.models.testUtil import check_layers_multitask


def test_AuxillaryLearning_A_trains():
    num_classes = 196
    num_makes = 1
    num_models = 1
    num_submodels = 1
    num_generations = 1
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = AuxillaryLearning_A(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)

def test_AuxillaryLearning_B_trains():
    num_classes = 196
    num_makes = 1
    num_models = 1
    num_submodels = 1
    num_generations = 1
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = AuxillaryLearning_B(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)


def test_AuxillaryLearning_C_trains():
    num_classes = 196
    num_makes = 1
    num_models = 1
    num_submodels = 1
    num_generations = 1
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = AuxillaryLearning_C(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)
    

    

