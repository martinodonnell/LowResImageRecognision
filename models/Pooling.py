import torch.nn as nn
import torch
class ChannelPoolingNetwork(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        print("Creating ChannelPoolingNetwork Model")

        self.base = base
        
        self.base.features = nn.Sequential(
            self.base.features,
            ChannelPool(compression=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )        
                
        self.base.classifier[0] =  nn.Linear(12544, 4096)
        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

       

        

    def forward(self, x):
        fc = self.base(x)
        return fc

class ChannelPool(nn.MaxPool1d):

    def __init__(self,compression, stride, padding, dilation, ceil_mode):
        super().__init__(stride, padding, dilation, ceil_mode)
        self.compression = compression


    def forward(self, input):
        n, c, w, h = input.size()
        c = int(c/self.compression)
        output = torch.zeros(n, c, w, h)

        index_pos = 0
        #Compress input to output tensor
        for index in range(0,input.size()[1],self.compression):
            output[0][index_pos] = torch.max(input[0][index],input[0][index+1])     
            index_pos +=1 
        return output


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
