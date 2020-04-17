import torchvision
import sys
sys.path.append("..")
from models.mtLearningClassic import MTLC_Shared_FC,MTLC_Shared_FC_B,MTLC_Shared_FC_C,MTLC_Seperate_FC,Old_MTLC_Seperate_FC
from unitTests.models.testUtil import check_layers_multitask



def test_MTLC_Shared_FC_A_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLC_Shared_FC(base,196, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)

def test_MTLC_Shared_FC_B_A_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLC_Shared_FC_B(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)

def test_MTLC_Shared_FC_C_A_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLC_Shared_FC_C(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)

def test_MTLC_Seperate_FC_A_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLC_Seperate_FC(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)

def test_Old_MTLC_Seperate_FC_A_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = Old_MTLC_Seperate_FC(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    check_layers_multitask(model)
