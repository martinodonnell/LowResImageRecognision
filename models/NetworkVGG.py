import torch.nn as nn
import torch


class NetworkV1(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__() #Running initialisation from super(NN.module)

        self.base = base

        in_features = self.base.classifier[6].in_features

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),

        )


    def forward(self, x):
        fc = self.base(x)
        return fc

class NetworkV1_1(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__() #Running initialisation from super(NN.module)


        self.base = base

        self.base.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            # nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.base.classifier[5] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            # nn.ReLU(),
            nn.Dropout(0.5),
        )

        in_features = self.base.classifier[6].in_features

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),

        )

    def forward(self, x):
        fc = self.base(x)
        return fc

class NetworkV1_2(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__() #Running initialisation from super(NN.module)

        self.base = base

        self.base.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            # nn.ReLU(),
            nn.Dropout(0.5),
        )

        in_features = self.base.classifier[6].in_features

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),

        )

    def forward(self, x):
        fc = self.base(x)
        return fc

class NetworkV1_3(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__() #Running initialisation from super(NN.module)


        self.base = base


        self.base.classifier[5] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            # nn.ReLU(),
            nn.Dropout(0.5),
        )

        in_features = self.base.classifier[6].in_features

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )
        
    def forward(self, x):
        fc = self.base(x)
        return fc

class NetworkV1_4(nn.Module):#http://cs230.stanford.edu/projects_spring_2019/reports/18681590.pdf
    def __init__(self, base, num_classes):#Define the layers
        super().__init__()

        self.base = base
        print(base)
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=25088,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=25088, out_features=512,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        

    def forward(self, x):
        fc = self.base(x)
        return fc

class NetworkV1_5(nn.Module):#http://cs230.stanford.edu/projects_spring_2019/reports/18681590.pdf Take 2
    def __init__(self, base, num_classes):#Define the layers
        super().__init__()

        self.base = base
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=512,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes),
        )

        

    def forward(self, x):
        fc = self.base(x)
        return fc


# class NetworkV2(nn.Module):
#     def __init__(self, base, num_classes,num_makes,num_models,num_submodels):
#         super().__init__() #Running initialisation from super(NN.module)

#         self.base = base

#         in_features = self.base.classifier[6].in_features
#         self.base.classifier[6] = nn.Sequential()

#         self.make_fc = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(in_features, num_makes)
#         )

#         self.model_fc = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(in_features, num_models)
#         )

#         self.submodel_fc = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(in_features, num_submodels)
#         )

#         self.class_fc = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.Linear(in_features + num_makes + num_models+ num_submodels, num_classes)
#         )        

#     def forward(self, x):
#         out = self.base(x)
#         make_fc = self.make_fc(out)
#         model_fc = self.model_fc(out)
#         submodel_fc = self.submodel_fc(out)

#         concat =  torch.cat([out,make_fc,model_fc,submodel_fc],dim=1)

#         fc = self.class_fc(concat)

        # return fc,make_fc,model_fc,submodel_fc

class NetworkV2(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_models,num_submodels):
        super().__init__()
        self.base = base

        in_features = self.base.classifier[6].in_features
        self.base.classifier[6] = nn.Sequential()

        self.brand_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_makes)
        )

        self.model_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_models)
        )

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features + num_makes + num_models, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        brand_fc = self.brand_fc(out)
        model_fc = self.model_fc(out)

        concat = torch.cat([out, brand_fc, model_fc], dim=1)

        fc = self.class_fc(concat)

        return fc, brand_fc, model_fc

class NetworkV2_Stan(nn.Module):
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
