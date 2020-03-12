import torch.nn as nn
import torch

#Model used in github repo ->https://github.com/JakubSochor/BoxCars/blob/master/scripts/train_eval.py
class NetworkV1(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__() #Running initialisation from super(NN.module)


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