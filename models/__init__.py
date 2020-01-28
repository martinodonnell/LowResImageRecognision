import torchvision
from models.NetworkVGG import NetworkV1,NetworkV2


# Set up config for other models in the future
def construct_model(config, num_classes,num_makes,num_models,num_submodels):
    base = torchvision.models.vgg16(pretrained=True)
    if config['model_version'] == 1:
        model = NetworkV1(base, num_classes)

    elif config['model_version'] == 2:
        model = NetworkV2(base, num_classes,num_makes,num_models,num_submodels)        
    else:
        print("Problem with version number in models")
        exit(1)

    return model
