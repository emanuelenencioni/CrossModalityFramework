from model.backbone import UnimodalBackbone
from model.yolox_head import YOLOXHead
import torch
import torch.nn as nn

class Detector(nn.Module):
    """
    Un modello di object detection completo che combina un backbone
    e una YOLOXHead.

    Args:
        backbone_name (str): Nome del modello timm da usare come backbone.
        num_classes (int): Numero di classi per la detection.
        pretrained (bool): Se usare pesi pre-allenati per il backbone.
    """
    def __init__(self, backbone_name: str,img_size: int, num_classes: int, pretrained: bool = True, model_name: str = None):
        super().__init__()
        
        # 1. Init backbone to extract 3 feat lvls
        self.backbone = UnimodalBackbone(
            backbone=backbone_name,
            img_size=img_size,
            pretrained=pretrained,
            outputs=["preflatten_feat"],
            out_indices=(2, 3, 4) # Stages C3, C4, C5
        )
        
        if model_name is None:
            self.model_name = f"{backbone_name}_yolox"
        else:
            self.model_name = model_name
        
        feature_info = self.backbone.feature_info
        in_channels = [info['num_chs'] for info in feature_info]
        strides = [info['reduction'] for info in feature_info]
        
        print(f"Backbone '{backbone_name}' inizializzato.")
        print(f"  - Strides estratti: {strides}")
        print(f"  - Canali di input per la Head: {in_channels}")

        # 3. Inizializza la YOLOXHead con i parametri corretti
        self.head = YOLOXHead(
            num_classes=num_classes,
            in_channels=in_channels,
            strides=strides
        )

    def get_name(self): return self.model_name

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        """
        Esegue il forward pass completo.

        Args:
            x (torch.Tensor): Immagine di input.
            targets (torch.Tensor, optional): I ground truth per il training. Se None, il modello è in modalità inferenza.

        Returns:
            Se targets non è None (training): un dizionario di loss.
            Se targets è None (inferenza): le predizioni del modello.
        """
        features = self.backbone(x)

        # La head YOLOX gestisce internamente sia la logica di training che di inferenza
        return self.head(features["preflatten_feat"], labels=targets)
