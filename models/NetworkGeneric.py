import torch.nn as nn
import torch

#Model used in github repo ->https://github.com/JakubSochor/BoxCars/blob/master/scripts/train_eval.py
class BoxCarsBase(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        print("Creating Boxcar Base Model")

        self.base = base

        self.base.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        

    def forward(self, x):
        fc = self.base(x)
        return fc


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


class AlexnetBase(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        print("Creating Alexnet Base Model")

        self.base = base

        print(self.base)
        in_features = self.base.classifier[6].in_features

        self.base.classifier[6] = nn.Linear(in_features, num_classes)


    def forward(self, x):
        fc = self.base(x)
        return fc