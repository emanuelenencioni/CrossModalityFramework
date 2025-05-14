#import head
#from backbone import BackboneAdapter
import torch
import torch.nn as nn
from torchvision import models


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackboneAdapter(models.resnet18(),400, 3)


    def forward(self, x):
        return self.backbone(x)

    def get_backbone(self):
        return self.backbone


if __name__=="__main__":
    model = model()
    print(model(x).size())
