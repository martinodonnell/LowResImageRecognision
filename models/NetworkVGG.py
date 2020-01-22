import torch.nn as nn


class NetworkV1(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__() #Running initialisation from super(NN.module)
        self.base = base

        in_features = self.base.classifier[6].in_features

        self.base.classifier[6] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        fc = self.base(x)
        return fc
