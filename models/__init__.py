import torchvision
from models.generic_networks import BoxCarsBase,StanfordBase,AlexnetBase
from models.multitask_networks import MTLC_Shared_FC,MTLC_Seperate_FC,Old_MTLC_Seperate_FC,MTLC_Shared_FC_B,MTLC_Shared_FC_C,MTLC_Stanford
from models.old_auxiliary_networks import MTLNC_Shared_FC_B,MTLNC_Seperate_FC,MTLNC_Shared_FC_A,MTLNC_Shared_FC_C
from models.pooling_networks import ChannelPoolingNetwork,SpatiallyWeightedPoolingNetwork
from models.auxillary_learning import AuxillaryLearning_A,AuxillaryLearning_B,AuxillaryLearning_C,AuxillaryLearning_A_B,AuxillaryLearning_B_B,AuxillaryLearning_C_B
from exceptions.exceptions import InvalidModelVersion

# Set up config for other models in the future
def construct_model(model_version, num_classes,num_makes,num_models,num_submodels,num_generation):
    base = torchvision.models.vgg16(pretrained=True, progress=True) 

    if model_version == 1:
        model = BoxCarsBase(base, num_classes)
    elif model_version == 2:
        model = MTLC_Shared_FC(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 3:
        model = MTLC_Seperate_FC(base, num_classes, num_makes, num_models,num_submodels,num_generation)    
    elif model_version == 4:
        model = Old_MTLC_Seperate_FC(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 5:
        model = MTLNC_Shared_FC_B(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 6:
        model = MTLNC_Seperate_FC(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 7:
        model = StanfordBase(base, num_classes)
    elif model_version == 8:
        model = ChannelPoolingNetwork(base, num_classes)
    elif model_version == 9:
        model = SpatiallyWeightedPoolingNetwork(base, num_classes)
    elif model_version == 10:
        model = MTLC_Shared_FC_B(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 11:
        model = MTLC_Shared_FC_C(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 12:
        model = MTLNC_Shared_FC_A(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 13:
        model = MTLNC_Shared_FC_C(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 14:
        base = torchvision.models.alexnet(pretrained=True, progress=True) 
        model = AlexnetBase(base, num_classes)
    elif model_version == 15:
        model = AuxillaryLearning_A(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 16:
        model = AuxillaryLearning_B(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 17:
        model = AuxillaryLearning_C(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 18:
        model = AuxillaryLearning_A_B(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 19:
        model = AuxillaryLearning_B_B(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 20:
        model = AuxillaryLearning_C_B(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    elif model_version == 21:
        model = MTLC_Stanford(base,num_classes,num_makes,num_models)
    else:
        raise InvalidModelVersion(str(model_version) + " is not a valid model version")

    return model
