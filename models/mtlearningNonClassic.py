import torch.nn as nn
import torch

#Same Boxcars fc layer
class MTLNC_Shard_FC(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels,num_generation):
        super().__init__()
        print("Creating non-classic single fc model")
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

        self.class_fc = nn.Sequential(
            nn.ReLU(),
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