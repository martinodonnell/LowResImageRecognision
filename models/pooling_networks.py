import torch.nn as nn
import torch
import time
class ChannelPoolingNetwork(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        print("Creating ChannelPoolingNetwork Model")

        self.base = base
        
        self.base.features = nn.Sequential(
            self.base.features,
            #Currently only works for this configuration
            ChannelPoolLayer(kernel_size=7, stride=2, padding=3, dilation=1, ceil_mode=False)
        )        

        self.base.classifier[0] =  nn.Linear(12544, 4096)
        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )  



    def forward(self, x):
        fc = self.base(x)
        return fc

class ChannelPoolLayer(nn.Module):

    def __init__(self, kernel_size=7, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.compression = 2
        self.output = None


    def forward(self, input):

        n, c, w, h = input.size()
        #Add padding to input so work with kernal size
        input = torch.nn.functional.pad(input, (0, 0, 0, 0, self.padding, self.padding), "constant", 0)
        
        #Get output
        # output = torch.empty(n, int(c/self.compression), w, h)
        # for x in range(n):
        output = torch.stack([ 
                        torch.stack(
                            [torch.max(input[x][index:index+self.kernel_size-1],axis=0)[0] #Get max from kernal size
                            for index in range(0,input.size()[1]-self.kernel_size,self.stride)]) #Move stride
                            for x in range(n)]) #Do work for each image in batch

        return output.cuda()

class SpatiallyWeightedPoolingNetwork(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        print("This network is still under contruction. Select a new one")
        exit()
        # print("Creating SpatiallyWeightedPooling Model")

        # self.base = base
        
        # self.base.features = nn.Sequential(
        #     self.base.features,
        #     SpatiallyWeightedPooling(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # )

        # fc_num_neurons = 512

        # self.base.classifier = nn.Sequential(
        #     nn.Linear(25088, fc_num_neurons),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(fc_num_neurons, fc_num_neurons),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(fc_num_neurons, num_classes)
        # )
        

    # def forward(self, x):
        # fc = self.base(x)
        # return fc
        

# class SpatiallyWeightedPooling(nn.MaxPool1d):
#     def forward(self, input):
#         n, c, w, h = input.size()
#         input = input.view(n,c,w*h).permute(0,2,1)
#         pooled =  nn.MaxPool1d(input)
#         _, _, c = input.size()
#         input = input.permute(0,2,1)
#         return input.view(n,c,w,h)
