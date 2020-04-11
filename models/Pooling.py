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
            ChannelPool(kernel_size=7, stride=2, padding=3, dilation=1, ceil_mode=False)
        )        

        self.base.classifier[0] =  nn.Linear(12544, 4096)
        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )  



    def forward(self, x):
        fc = self.base(x)
        return fc

# class ChannelPool(nn.Module):


#     def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
#                  return_indices=False, ceil_mode=False):
#         super().__init__()

#         self.kernel_size = kernel_size
#         self.stride = stride or kernel_size
#         self.padding = padding
#         self.dilation = dilation
#         self.return_indices = return_indices
#         self.ceil_mode = ceil_mode

#         self.compression = 2#Hard coded but can be calcualted

#     def foward(self,input):
#         #Get output tensor dimensions depending on compressions
#         print(1)
#         n, c, w, h = input.size()
#         print(2)
#         c = int(c/self.compression)
#         # output = torch.empty(n, c, w, h).cuda()
#         output = torch.zeros(n, c, w, h)
#         print(3)

#         #Add padding to input so work with kernal size
#         input = torch.nn.functional.pad(input, (0, 0, 0, 0, self.padding, self.padding), "constant", 0)

#         index_pos = 0
#         count = 1
#         #Compress input to output tensor
#         for index in range(0,input.size()[1]-self.stride):
#             pooled_channels = input[0][index]
#             for x in range(1,self.kernal_size):
#                 pooled_channels = torch.max(pooled_channels,input[0][index+x]) 
            
#             output[0][index_pos] = pooled_channels
#             index_pos +=1
#         return output

class ChannelPool(nn.Module):

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
        #Add padding to input so work with kernal size
        input = torch.nn.functional.pad(input, (0, 0, 0, 0, self.padding, self.padding), "constant", 0)
        
        #Get output
        output = torch.empty(1,256,7,7).cuda()
        output[0] = torch.stack([torch.max(input[0][index:index+self.kernel_size-1],axis=0)[0]
                  for index in range(0,input.size()[1]-self.kernel_size,self.stride)])

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
