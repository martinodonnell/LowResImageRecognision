import torch.nn as nn
import torch
class ChannelPoolingNetwork(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        print("Creating ChannelPoolingNetwork Model")

        self.base = base
        
        self.base.features = nn.Sequential(
            self.base.features,
            ChannelPool(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        print(self.base.features)
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

class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled =  nn.MaxPool1d(input)
        _, _, c = input.size()
        input = input.permute(0,2,1)
        return input.view(n,c,w,h)


class SpatiallyWeightedPoolingNetwork(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        print("Creating SpatiallyWeightedPooling Model")

        self.base = base
        
        self.base.features = nn.Sequential(
            self.base.features,
            SpatiallyWeightedPooling(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        fc_num_neurons = 512

        self.base.classifier = nn.Sequential(
            nn.Linear(25088, fc_num_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_num_neurons, fc_num_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_num_neurons, num_classes)
        )

        print(self.base)
        exit()


        

    def forward(self, x):
        fc = self.base(x)
        return fc
        

class SpatiallyWeightedPooling(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled =  nn.MaxPool1d(input)
        _, _, c = input.size()
        input = input.permute(0,2,1)
        return input.view(n,c,w,h)
