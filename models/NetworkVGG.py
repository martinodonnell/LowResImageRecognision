import torch.nn as nn
import torch


class NetworkV1(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__() #Running initialisation from super(NN.module)

        # base = torchvision.models.vgg16(pretrained=True)

        self.base = base

        in_features = self.base.classifier[6].in_features

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        fc = self.base(x)
        return fc


class NetworkV2(nn.Module):
    def __init__(self, base, num_classes,num_makes,num_models,num_submodels):
        super().__init__() #Running initialisation from super(NN.module)

        # base = torchvision.models.vgg16(pretrained=True)
        self.base = base

        print(self.base)
        in_features = self.base.classifier.in_features
        self.base.classifier = nn.Sequential()

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

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features + num_makes + num_models+ num_submodels, num_classes)
        )
        

    def forward(self, x):
        out = self.base(x)
        make_fc = self.make_fc
        model_fc = self.model_fc
        submodel_fc = self.submodel_fc

        concat =  torch.cat([out,make_fc,model_fc,submodel_fc],dim=1)

        fc = self.class_fc(concat)

        return fc,make_fc,model_fc,submodel_fc
