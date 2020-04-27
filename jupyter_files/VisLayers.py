



from mirror import mirror
from mirror.visualisations.web import *
from PIL import Image
from torchvision.models import vgg16
from torchvision.transforms import ToTensor, Resize, Compose
import torch.nn as nn
import torch



class StanfordBase(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        print("Creating Stanford Base Model")

        self.base = base

        in_features = self.base.classifier[6].in_features

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),

        )


    def forward(self, x):
        fc = self.base(x)
        return fc

def load_weight(model, path, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)



# read in model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base = vgg16(pretrained=True, progress=True) 
net = StanfordBase(base, 107)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_weight(net,'/Users/martinodonnell/Documents/uni/fourth_year/4006/LowResImageRecognision/saves/342_model.pth',device)

net = net.to(device)
final_conv = 'base'

# open some images
car_1 = Image.open("../data/BoxCars/images/001/0/019700_000.png")
car_2 = Image.open("../data/BoxCars/images/104/11/026411_001.png")
# resize the image and make it a tensor
to_input = Compose([Resize((224, 224)), ToTensor()])
# call mirror with the inputs and the model
mirror([to_input(car_1),to_input(car_2)], net, visualisations=[BackProp, GradCam, DeepDream])