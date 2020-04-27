import torchvision
import sys
sys.path.append("..")
from models.auxillary_learning import AuxillaryLearning_A,AuxillaryLearning_B,AuxillaryLearning_C,AuxillaryLearning_A_B,AuxillaryLearning_B_B,AuxillaryLearning_C_B
from unitTests.models.test_util import check_layers_multitask,check_classes_in_output
    
def test_AuxillaryLearning_A_trains():
    num_classes = 196
    num_makes = 2
    num_models = 2
    num_submodels = 2
    num_generations = 2

    #Check model is learning
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = AuxillaryLearning_A(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    output = check_layers_multitask(model)

    #Ensure the correct number of outputs in each model iss output depending on variables above
    check_classes_in_output(output,[num_classes,num_makes,num_models,num_submodels,num_generations])




def test_AuxillaryLearning_B_trains():
    num_classes = 196
    num_makes = 1
    num_models = 1
    num_submodels = 1
    num_generations = 1

    #Check model is learning
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = AuxillaryLearning_B(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    output = check_layers_multitask(model)

    #Ensure the correct number of outputs in each model iss output depending on variables above
    check_classes_in_output(output,[num_classes,num_makes,num_models,num_submodels,num_generations])


def test_AuxillaryLearning_C_trains():
    num_classes = 196
    num_makes = 1
    num_models = 1
    num_submodels = 1
    num_generations = 1
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = AuxillaryLearning_C(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    output = check_layers_multitask(model)

    #Ensure the correct number of outputs in each model iss output depending on variables above
    check_classes_in_output(output,[num_classes,num_makes,num_models,num_submodels,num_generations])


def test_AuxillaryLearning_A_B_trains():
    num_classes = 196
    num_makes = 1
    num_models = 1
    num_submodels = 1
    num_generations = 1
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = AuxillaryLearning_A_B(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    output = check_layers_multitask(model)

    #Ensure the correct number of outputs in each model iss output depending on variables above
    check_classes_in_output(output,[num_classes,num_makes,num_models,num_submodels,num_generations])


def test_AuxillaryLearning_B_B_trains():
    num_classes = 196
    num_makes = 1
    num_models = 1
    num_submodels = 1
    num_generations = 1
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = AuxillaryLearning_B(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    output = check_layers_multitask(model)

    #Ensure the correct number of outputs in each model iss output depending on variables above
    check_classes_in_output(output,[num_classes,num_makes,num_models,num_submodels,num_generations])


def test_AuxillaryLearning_C_B_trains():
    num_classes = 196
    num_makes = 1
    num_models = 1
    num_submodels = 1
    num_generations = 1
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = AuxillaryLearning_C_B(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    output = check_layers_multitask(model)

    #Ensure the correct number of outputs in each model iss output depending on variables above
    check_classes_in_output(output,[num_classes,num_makes,num_models,num_submodels,num_generations])
    

    

