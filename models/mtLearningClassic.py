import torch.nn as nn
import torch

#Classic MT Learning but with boxcars fc layer
class MTLC_Shared_FC(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating classic single fc model")

        self.base = base

        self.base.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        in_features = 4096
        self.make_fc = nn.Sequential(
            nn.Linear(in_features, num_makes)
        )

        self.model_fc = nn.Sequential(
            nn.Linear(in_features, num_models)
        )

        self.submodel_fc = nn.Sequential(
            nn.Linear(in_features, num_submodels)
        )

        self.generation_fc = nn.Sequential(
            nn.Linear(in_features, num_generation)            
        )

    def forward(self, x):
        out = self.base(x)
        
        make_fc = self.make_fc(out)
        model_fc = self.model_fc(out)
        submodel_fc = self.submodel_fc(out)
        generation_fc = self.generation_fc(out)

        return make_fc, model_fc,submodel_fc,generation_fc

#Classic MT Learning but boxcars each feature has own fc layer
class MTLC_Seperate_FC(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating classic seperate feature fc model")

        self.base = base

        #Remove classifier from vgg16
        self.base.classifier = nn.Sequential(
        )

        int_features = 4096
        self.make_fc = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int_features, num_makes)
        )

        self.model_fc = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int_features, num_models)
        )

        self.submodel_fc = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int_features, num_submodels)
        )

        self.generation_fc = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int_features, num_generation)            
        )

    def forward(self, x):
        out = self.base(x)
        
        make_fc = self.make_fc(out)
        model_fc = self.model_fc(out)
        submodel_fc = self.submodel_fc(out)
        generation_fc = self.generation_fc(out)

        return make_fc, model_fc,submodel_fc,generation_fc


#Classic multitask learning. Pass fector vector from CNN/Some fc to each feacture. Then get prediction
class Old_MTLC_Seperate_FC(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        self.base = base

        in_features = self.base.classifier[6].in_features
        self.base.classifier[6] = nn.Sequential()

        self.make_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_makes)
        )

        self.model_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_models)
        )

        self.submodel_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_submodels)
        )

        self.generation_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_generation)
        )

        
    def forward(self, x):
        out = self.base(x)
        
        make_fc = self.make_fc(out)
        model_fc = self.model_fc(out)
        submodel_fc = self.submodel_fc(out)
        generation_fc = self.generation_fc(out)

        return make_fc, model_fc,submodel_fc,generation_fc