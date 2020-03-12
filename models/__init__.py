import torchvision
from models.NetworkGeneric import NetworkV1 as Base_Model 
from models.mtLearningClassic import MTLC_Shared_FC,MTLC_Seperate_FC,Old_MTLC_Seperate_FC
from models.mtLearningNonClassic import MTLNC_Shard_FC 

# Set up config for other models in the future
def construct_model(config, num_classes,num_makes,num_models,num_submodels,num_generation):
    base = torchvision.models.vgg16(pretrained=True, progress=True)   

    if config['model_version'] == 1:
        model = Base_Model(base, num_classes)

    elif config['model_version'] == 2:
        model = MTLC_Shared_FC(base, num_classes, num_makes, num_models,num_submodels,num_generation)

    elif config['model_version'] == 3:
        model = MTLC_Seperate_FC(base, num_classes, num_makes, num_models,num_submodels,num_generation)
    
    elif config['model_version'] == 4:
        model = Old_MTLC_Seperate_FC(base, num_classes, num_makes, num_models,num_submodels,num_generation)

    elif config['model_version'] == 5:
        model = MTLNC_Shard_FC(base, num_classes, num_makes, num_models,num_submodels,num_generation)

    return model
