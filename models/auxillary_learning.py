import torch.nn as nn
import torch


class AuxillaryLearning_A(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating Auxillary Model A")
        self.base = base

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5)
        )
        
        in_features = 4096
        self.class_fc_one = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

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

        self.class_fc = nn.Sequential(
            nn.Linear(num_classes + num_makes + num_models+num_submodels+num_generation, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        class_fc_one_out = self.class_fc_one(out)
        make_out = self.make_fc(out)
        model_out = self.model_fc(out)
        submodel_out = self.submodel_fc(out)
        generation_out = self.generation_fc(out)

        concat = torch.cat([class_fc_one_out, make_out, model_out,submodel_out,generation_out], dim=1)

        fc = self.class_fc(concat)

        return fc, make_out, model_out,submodel_out,generation_out

#Same Boxcars fc layer
class AuxillaryLearning_B(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating Auxillary Model B")
        self.base = base

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5)
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

        self.class_fc = nn.Sequential(
            nn.Linear(in_features + num_makes + num_models+num_submodels+num_generation, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        make_out = self.make_fc(out)
        model_out = self.model_fc(out)
        submodel_out = self.submodel_fc(out)
        generation_out = self.generation_fc(out)

        concat = torch.cat([out, make_out, model_out,submodel_out,generation_out], dim=1)

        fc = self.class_fc(concat)

        return fc, make_out, model_out,submodel_out,generation_out



#Same Boxcars fc layer
class AuxillaryLearning_C(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating Auxillary Model C")
        self.base = base

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5)
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

        self.class_fc = nn.Sequential(
            nn.Linear(num_makes + num_models+num_submodels+num_generation, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        
        make_out = self.make_fc(out)
        model_out = self.model_fc(out)
        submodel_out = self.submodel_fc(out)
        generation_out = self.generation_fc(out)

        concat = torch.cat([make_out, model_out,submodel_out,generation_out], dim=1)

        fc = self.class_fc(concat)

        return fc, make_out, model_out,submodel_out,generation_out



# ----------------------------------------------------------------------
# 
# 
# 
# Following structure from Model B in multitask learning
# 
# 
# ----------------------------------------------------------------------


class AuxillaryLearning_A_B(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating Auxillary Model D")
        self.base = base

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5)
        )
        
        in_features = 4096
        self.class_fc_one = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

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

        self.class_fc = nn.Sequential(
            nn.Linear(num_classes + num_makes + num_models+num_submodels+num_generation, num_classes)
        )

    def forward(self, x):
        
        #Pass thought vgg16 network
        out = self.base(x)

        #Firts output for fine-grain label
        class_fc_one_out = self.class_fc_one(out)

        #Task hierachys process
        make_fc = self.make_fc(out)
        concat = torch.cat([out, make_fc], dim=1)
        model_fc = self.model_fc(concat)
        concat = torch.cat([out,model_fc], dim=1)
        submodel_fc = self.submodel_fc(concat)
        concat = torch.cat([out,submodel_fc], dim=1)
        generation_fc = self.generation_fc(concat)


        concat = torch.cat([class_fc_one_out, make_fc, model_fc,submodel_fc,generation_fc], dim=1)

        #Calculate full fine-grain label
        fc = self.class_fc(concat)

        return fc, make_fc, model_fc,submodel_fc,generation_fc

#Same Boxcars fc layer
class AuxillaryLearning_B_B(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating Auxillary Model E")
        self.base = base

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5)
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

        self.class_fc = nn.Sequential(
            nn.Linear(in_features + num_makes + num_models+num_submodels+num_generation, num_classes)
        )

    def forward(self, x):
        out = self.base(x)

        #Task hierachys process
        make_fc = self.make_fc(out)
        concat = torch.cat([out, make_fc], dim=1)
        model_fc = self.model_fc(concat)
        concat = torch.cat([out,model_fc], dim=1)
        submodel_fc = self.submodel_fc(concat)
        concat = torch.cat([out,submodel_fc], dim=1)
        generation_fc = self.generation_fc(concat)

        concat = torch.cat([out, make_fc, model_fc,submodel_fc,generation_fc], dim=1)

        #Calculate full fine-grain label
        fc = self.class_fc(concat)

        return fc, make_fc, model_fc,submodel_fc,generation_fc



#Same Boxcars fc layer
class AuxillaryLearning_C_B(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating Auxillary Model F")
        self.base = base

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5)
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

        self.class_fc = nn.Sequential(
            nn.Linear(num_makes + num_models+num_submodels+num_generation, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        
        #Task hierachys process
        make_fc = self.make_fc(out)
        concat = torch.cat([out, make_fc], dim=1)
        model_fc = self.model_fc(concat)
        concat = torch.cat([out,model_fc], dim=1)
        submodel_fc = self.submodel_fc(concat)
        concat = torch.cat([out,submodel_fc], dim=1)
        generation_fc = self.generation_fc(concat)

        concat = torch.cat([make_fc, model_fc,submodel_fc,generation_fc], dim=1)

        fc = self.class_fc(concat)

        return fc, make_fc, model_fc,submodel_fc,generation_fc
