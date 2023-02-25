import timm
import torch
import torch.nn as nn

class HeroModel(nn.Module):
    """ The model uses a backbone classification from timm """
    def __init__(self, backbone_name):
        super().__init__()
        
        # Load backbone from timm
        self.backbone = timm.create_model(model_name = backbone_name, pretrained = True, num_classes = 64)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        return self.softmax(self.backbone(x))