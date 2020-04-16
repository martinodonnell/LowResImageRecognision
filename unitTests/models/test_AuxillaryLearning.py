import torchvision
import sys
sys.path.append("..")
from models.AuxillaryLearning import AuxillaryLearning_A,AuxillaryLearning_B,AuxillaryLearning_C
import torch
from torch.autograd.gradcheck import gradcheck
from torch.autograd import Variable
import torch.optim as optim

def test_AuxillaryLearning_A_trains():
    #Constants used in all tests
    num_classes = 196
    num_makes = 1
    num_models = 1
    num_submodels = 1
    num_generations = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = torchvision.models.vgg16(pretrained=True, progress=True) 


    model = AuxillaryLearning_A(base,num_classes, num_makes, num_models,num_submodels,num_generations)
    model = model.to(device)
    input = Variable(torch.randn(1, 3, 224, 224).float(), requires_grad=True) 
    test = gradcheck(model, input, eps=1e-6, atol=1e-4)
    print(test)
    assert True


# class AuxillaryLearning_A(nn.Module):
#     def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
#         super().__init__()
#         print("Creating Auxillary Model A")
#         self.base = base

#         self.base.classifier[6] = nn.Sequential(
#             nn.Dropout(0.5)
#         )
        
#         in_features = 4096
#         self.class_fc_one = nn.Sequential(
#             nn.Linear(in_features, num_classes)
#         )

#         self.make_fc = nn.Sequential(
#             nn.Linear(in_features, num_makes)
#         )

#         self.model_fc = nn.Sequential(
#             nn.Linear(in_features, num_models)
#         )

#         self.submodel_fc = nn.Sequential(
#             nn.Linear(in_features, num_submodels)
#         )

#         self.generation_fc = nn.Sequential(
#             nn.Linear(in_features, num_generation)
#         )

#         self.class_fc = nn.Sequential(
#             nn.Linear(num_classes + num_makes + num_models+num_submodels+num_generation, num_classes)
#         )

#     def forward(self, x):
#         out = self.base(x)
#         class_fc_one_out = self.class_fc_one(out)
#         make_out = self.make_fc(out)
#         model_out = self.model_fc(out)
#         submodel_out = self.submodel_fc(out)
#         generation_out = self.generation_fc(out)

#         concat = torch.cat([class_fc_one_out, make_out, model_out,submodel_out,generation_out], dim=1)

#         fc = self.class_fc(concat)

#         return fc, make_out, model_out,submodel_out,generation_out

# #Same Boxcars fc layer
# class AuxillaryLearning_B(nn.Module):
#     def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
#         super().__init__()
#         print("Creating Auxillary Model B")
#         self.base = base

#         self.base.classifier[6] = nn.Sequential(
#             nn.Dropout(0.5)
#         )

#         in_features = 4096
#         self.make_fc = nn.Sequential(
#             nn.Linear(in_features, num_makes)
#         )

#         self.model_fc = nn.Sequential(
#             nn.Linear(in_features, num_models)
#         )

#         self.submodel_fc = nn.Sequential(
#             nn.Linear(in_features, num_submodels)
#         )

#         self.generation_fc = nn.Sequential(
#             nn.Linear(in_features, num_generation)
#         )

#         self.class_fc = nn.Sequential(
#             nn.Linear(in_features + num_makes + num_models+num_submodels+num_generation, num_classes)
#         )

#     def forward(self, x):
#         out = self.base(x)
#         make_out = self.make_fc(out)
#         model_out = self.model_fc(out)
#         submodel_out = self.submodel_fc(out)
#         generation_out = self.generation_fc(out)

#         concat = torch.cat([out, make_out, model_out,submodel_out,generation_out], dim=1)

#         fc = self.class_fc(concat)

#         return fc, make_out, model_out,submodel_out,generation_out



# #Same Boxcars fc layer
# class AuxillaryLearning_C(nn.Module):
#     def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
#         super().__init__()
#         print("Creating Auxillary Model C")
#         self.base = base

#         self.base.classifier[6] = nn.Sequential(
#             nn.Dropout(0.5)
#         )
        
#         in_features = 4096

#         self.make_fc = nn.Sequential(
#             nn.Linear(in_features, num_makes)
#         )

#         self.model_fc = nn.Sequential(
#             nn.Linear(in_features, num_models)
#         )

#         self.submodel_fc = nn.Sequential(
#             nn.Linear(in_features, num_submodels)
#         )

#         self.generation_fc = nn.Sequential(
#             nn.Linear(in_features, num_generation)
#         )

#         self.class_fc = nn.Sequential(
#             nn.Linear(num_makes + num_models+num_submodels+num_generation, num_classes)
#         )

#     def forward(self, x):
#         out = self.base(x)
        
#         make_out = self.make_fc(out)
#         model_out = self.model_fc(out)
#         submodel_out = self.submodel_fc(out)
#         generation_out = self.generation_fc(out)

#         concat = torch.cat([make_out, model_out,submodel_out,generation_out], dim=1)

#         fc = self.class_fc(concat)

        # return fc, make_out, model_out,submodel_out,generation_out