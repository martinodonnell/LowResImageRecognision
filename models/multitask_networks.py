import torch.nn as nn
import torch

#Classic MT Learning but with boxcars fc layer
class MTLC_Shared_FC(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating classic MTL single fc model")

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

#Classic MT Learning but with boxcars fc layer
class MTLC_Shared_FC_B(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating classic MTL single fc model")

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
            nn.Linear(in_features+num_makes, num_models)
        )

        self.submodel_fc = nn.Sequential(
            nn.Linear(in_features+num_models, num_submodels)
        )

        self.generation_fc = nn.Sequential(
            nn.Linear(in_features+num_submodels, num_generation)            
        )

    def forward(self, x):
        out = self.base(x)        
        make_fc = self.make_fc(out)

        concat = torch.cat([out, make_fc], dim=1)
        model_fc = self.model_fc(concat)

        concat = torch.cat([out,model_fc], dim=1)
        submodel_fc = self.submodel_fc(concat)

        concat = torch.cat([out,submodel_fc], dim=1)
        generation_fc = self.generation_fc(concat)

        return make_fc, model_fc,submodel_fc,generation_fc

#Classic MT Learning but with boxcars fc layer
class MTLC_Shared_FC_C(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating classic MTL single fc model")

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
            nn.Linear(num_makes, num_models)
        )

        self.submodel_fc = nn.Sequential(
            nn.Linear(num_models, num_submodels)
        )

        self.generation_fc = nn.Sequential(
            nn.Linear(num_submodels, num_generation)            
        )

    def forward(self, x):
        out = self.base(x)
        
        make_fc = self.make_fc(out)
        model_fc = self.model_fc(make_fc)
        submodel_fc = self.submodel_fc(model_fc)
        generation_fc = self.generation_fc(submodel_fc)
        return make_fc, model_fc,submodel_fc,generation_fc

#Classic MT Learning but boxcars each feature has own fc layer
class MTLC_Seperate_FC(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating classic MTL seperate feature fc model")

        self.base = base

        #Remove classifier from vgg16
        self.base.classifier = nn.Sequential(
        )

        in_features = 4096
        self.make_fc = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_makes)
        )

        self.model_fc = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_models)
        )

        self.submodel_fc = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_submodels)
        )

        self.generation_fc = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_generation)            
        )

    def forward(self, x):
        out = self.base(x)
        
        make_fc = self.make_fc(out)
        model_fc = self.model_fc(out)
        submodel_fc = self.submodel_fc(out)
        generation_fc = self.generation_fc(out)

        return make_fc, model_fc,submodel_fc,generation_fc


#Old version taht I ran test 100-17 on. This is the old way of creating the fc layers
class Old_MTLC_Seperate_FC(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating classic MTL seperate feature fc model(OLD FC Version)")
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


class MTLC_Stanford(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_types):
        super().__init__()
        self.base = base

        in_features = self.base.classifier[6].in_features
        self.base.classifier[6] = nn.Sequential()

        self.brand_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_makes)
        )

        self.type_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_types)
        )

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features + num_makes + num_types, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        brand_fc = self.brand_fc(out)
        type_fc = self.type_fc(out)

        concat = torch.cat([out, brand_fc, type_fc], dim=1)

        fc = self.class_fc(concat)

        return fc, brand_fc, type_fc