import torchvision
import sys
sys.path.append("..")

from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
from PIL import Image
from torchvision import transforms



class TestingRandomDatasetSingle(Dataset):   
    def __init__(self,num_samples):
        self.transform =  transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
                ]
        )    
        self.num_samples = num_samples 
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # inputs = Variable(torch.randn(3,244,244), requires_grad=True)
        img = Image.open('/Users/martinodonnell/Documents/uni/fourth_year/4006/LowResImageRecognision/data/BoxCars/images/001/0/019700_000.png')
        img = img.convert('RGB')
        img = self.transform(img)

        return img,0


class TestingRandomDatasetMulti(Dataset):   
    def __init__(self,num_samples):
        self.transform =  transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4706145, 0.46000465, 0.45479808), (0.26668432, 0.26578658, 0.2706199))
                ]
        )    
        self.num_samples = num_samples 
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # inputs = Variable(torch.randn(3,244,244), requires_grad=True)
        img = Image.open('/Users/martinodonnell/Documents/uni/fourth_year/4006/LowResImageRecognision/data/BoxCars/images/001/0/019700_000.png')
        img = img.convert('RGB')
        img = self.transform(img)

        return img,0,0,0,0,0


def compare_model_keys_same(model1,model2):
    for key1,key2 in zip(model1.state_dict(),model2.state_dict()):
        if key1!=key2:
            return False
    return True

def compare_model_keys_diff(model1,model2):
    for key1,key2 in zip(model1.state_dict(),model2.state_dict()):
        if key1==key2:
            return True
    return True