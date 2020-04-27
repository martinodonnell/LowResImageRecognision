import torchvision
import sys
sys.path.append("..")
from models.old_auxiliary_networks import MTLNC_Shared_FC_B,MTLNC_Seperate_FC,MTLNC_Shared_FC_A,MTLNC_Shared_FC_C
from unit_tests.models.test_util import check_layers_multitask,check_classes_in_output


def test_MTLNC_Shared_FC_B_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLNC_Shared_FC_B(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    output = check_layers_multitask(model)

    #Ensure the correct number of outputs in each model iss output depending on variables above
    check_classes_in_output(output,[num_classes,num_makes,num_models,num_submodels,num_generations])


def test_MTLNC_Seperate_FC_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLNC_Seperate_FC(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    output = check_layers_multitask(model)

    #Ensure the correct number of outputs in each model iss output depending on variables above
    check_classes_in_output(output,[num_classes,num_makes,num_models,num_submodels,num_generations])


def test_MTLNC_Shared_FC_A_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLNC_Shared_FC_A(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    output = check_layers_multitask(model)

    #Ensure the correct number of outputs in each model iss output depending on variables above
    check_classes_in_output(output,[num_classes,num_makes,num_models,num_submodels,num_generations])

def test_MTLNC_Shared_FC_C_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = MTLNC_Shared_FC_C(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    output = check_layers_multitask(model)

    #Ensure the correct number of outputs in each model iss output depending on variables above
    check_classes_in_output(output,[num_classes,num_makes,num_models,num_submodels,num_generations])