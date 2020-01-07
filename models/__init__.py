import torchvision
from models.NetworkVGG import NetworkV1


# Set up config for other models in the future
def construct_model(config, num_classes):
    base = torchvision.models.vgg16(pretrained=True)
    model = NetworkV1(base, num_classes)

    return model
