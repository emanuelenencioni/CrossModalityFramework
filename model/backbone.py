import torchvision.models as models
import torch
import torch.nn as nn


# TODO: gestire modelli esterni a torch -> dare linee guida per struttura basic.
class BackboneAdapter(nn.Module):
    """
    Initializes the Backbone model.
    Args:
        model (torchvision.models): A pre-trained torchvision model. (at least for now)
        pretrained (bool): If True, loads pre-trained weights for the model. Defaults to True.
        input_dim: squared dim for input frame
        input_ch: input channel
    """
    def __init__(self, model,input_dim, input_ch, torch_pretrained=False, custom_weights=""):
        super(Backbone, self).__init__()
        assert model != None, "Insert a valid model"
        if torch_pretrained:
            model_name = model.__class__.__name__.lower()
            print(model_name)
            try:
                
                self.backbone = getattr(models, model_name)(pretrained=True)
                
            except Exception as e:
                print(f"Error loading torch pretrained weights: {e}")
                self.backbone = model
        else:
            self.backbone = model

        if hasattr(self.backbone, 'fc'):
            self.model.fc = nn.Identity()
        if hasattr(self.backbone, 'classifier'):
            self.model.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = BackboneAdapter(models.resnet18(),512,3,True)
    print(model)
    x = torch.randn(1,3,224,224)
    print(model(x).size())
