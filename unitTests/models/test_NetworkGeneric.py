import torchvision
import sys
sys.path.append("..")
from models.NetworkGeneric import BoxCarsBase,StanfordBase,AlexnetBase
from unitTests.models.testUtil import check_layers_update


def test_BoxCarsBase_trains():
    num_classes = 196
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = BoxCarsBase(base,num_classes)
    check_layers_update(model)

def test_StanfordBase_trains():
    num_classes = 196
    base = torchvision.models.vgg16(pretrained=True, progress=True)     
    model = StanfordBase(base,num_classes)
    check_layers_update(model)

def test_AlexnetBase_trains():
    num_classes = 196
    base = torchvision.models.alexnet(pretrained=True, progress=True)    
    model = AlexnetBase(base,num_classes)
    check_layers_update(model)



