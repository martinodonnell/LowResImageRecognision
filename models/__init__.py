import torchvision
from models.NetworkVGG import NetworkV1,NetworkV1_1,NetworkV1_2,NetworkV1_3,NetworkV1_4,NetworkV2


# Set up config for other models in the future
def construct_model(config, num_classes,num_makes,num_models,num_submodels):
    base = torchvision.models.vgg16(pretrained=True, progress=True)   
    if config['model_version'] == 1:
        model = NetworkV1(base, num_classes)

    elif config['model_version'] == 2:
        model = NetworkV2(base, num_classes,num_makes,num_models,num_submodels)

    elif config['model_version'] == 3:
        model = NetworkV1_1(base, num_classes)

    elif config['model_version'] == 4:
        model = NetworkV1_2(base, num_classes)

    elif config['model_version'] == 5:
        model = NetworkV1_3(base, num_classes)

    elif config['model_version'] == 6:
        model = NetworkV1_4(base, num_classes)
    

    return model
