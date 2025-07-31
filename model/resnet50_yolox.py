from model.backbone import UnimodalBackbone
from model.yolox_head import YOLOXHead
import torch
import torch.nn as nn

class Resnet50_yolox(nn.Module):
    """
    Un modello di object detection completo che combina backbone resnet50 con una head YOLOX.

    Args:
        backbone_name (str): Nome del modello timm da usare come backbone.
        num_classes (int): Numero di classi per la detection.
        pretrained (bool): Se usare pesi pre-allenati per il backbone.
    """
    def __init__(self,  backbone: dict, name: str="resnet50_yolox", head: dict={'name': 'yolox_head','num_classes': 8}):
        super().__init__()
        assert 'output_indices' in backbone, "Error - output_indices must be specified in the backbone config"
        out_indices = backbone['output_indices']
        # 1. Init backbone to extract 3 feat lvls
        self.backbone = UnimodalBackbone(
            backbone=backbone.get('rgb_backbone', backbone.get('event_backbone', None)),
            pretrained=backbone.get('pretrained', True),
            pretrained_weights=backbone.get("pretrained_weights", None),
            img_size=backbone.get('input_size'),
            outputs=["preflatten_feat"],
            output_indices=backbone.get('output_indices')  # Default to last 3 layers if not specified
        )
        

        self.name = name
        
        feature_info = self.backbone.feature_info
        in_channels = [info['num_chs'] for info in feature_info if info['index'] in out_indices]
        strides = [info['reduction'] for info in feature_info if info['index'] in out_indices]
        
        print(f"Backbone '{backbone['name']}' inizializzato.")
        print(f"  - Strides estratti: {strides}")
        print(f"  - Canali di input per la Head: {in_channels}")
        
        # 3. Inizializza la YOLOXHead con i parametri corretti
        self.head = YOLOXHead(
            num_classes=head['num_classes'],
            in_channels=in_channels,
            strides=strides
        )

    def get_name(self): return self.name

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
