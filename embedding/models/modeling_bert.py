from typing import List
from .base_model import BaseBackboneWrapper, BaseEmbedder

class BERTBackboneWrapper(BaseBackboneWrapper):
    def __init__(self, 
                 backbone, 
                 pool_type: str = 'cls', 
                 checkpoint_batch_size: int = -1, 
                 which_layer: int = -1, 
                 reserved_layers: List[int]=None,
                 lora_config: bool = True, 
                 self_extend: bool = False,
                 **kwargs):
        # backbone = AutoModel.from_pretrained(backbone)
        super().__init__(backbone, 
                         pool_type, 
                         checkpoint_batch_size, 
                         which_layer, 
                         reserved_layers, 
                         lora_config, 
                         self_extend,
                         **kwargs)

    def _init_backbone(self, backbone, **kwargs):
        return super()._init_backbone(backbone, **kwargs)

    def partial_encode(self, *inputs):
        return super().partial_encode(*inputs)

    def backbone_embedding(self, input_ids):
        return super().backbone_embedding(input_ids)

    def backbone_forward(self, input_ids):
        return super().backbone_forward(input_ids)

    def lora_wrapper(self, model):
        return super().lora_wrapper(model)
    
    def model_razor(self, backbone, reserved_layers):
        for lay in range(backbone.config.num_hidden_layers)[::-1]:
            if lay not in reserved_layers:
                del(backbone.encoder.layer[lay])
        # reset config
        backbone.config.num_hidden_layers = len(backbone.encoder.layer)
        print('Current model layers: ', backbone.config.num_hidden_layers)    
        return backbone
    

class BertEmbedder(BaseEmbedder):
    def __init__(self, 
                 backbone: str, 
                 backbone_wrapper: BaseBackboneWrapper=BERTBackboneWrapper, 
                 pool_type: str = 'cls', 
                 checkpoint_batch_size=-1, 
                 flashatt: bool=False, 
                 which_layer: int = -1,
                 reserved_layers: list = None,
                 lora_config: bool = False, 
                 mytryoshka_indexes: list = None, 
                 normalize: bool = False,
                 **kwargs):
        super().__init__(backbone, 
                         backbone_wrapper, 
                         pool_type, 
                         checkpoint_batch_size, 
                         flashatt, 
                         which_layer, 
                         reserved_layers, 
                         lora_config, 
                         mytryoshka_indexes, 
                         normalize, 
                         **kwargs)
