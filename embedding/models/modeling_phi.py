import torch
from typing import Union
from peft import LoraConfig
from .base_model import BaseBackboneWrapper, BaseEmbedder

# 适用于causalLM？
class PhiBackboneWrapper(BaseBackboneWrapper):
    def __init__(self, backbone: str, pool_type: str='cls', checkpoint_batch_size: int=-1, which_layer: int=-1, lora_config: Union[bool, LoraConfig]=False, self_extend: bool=True):
        super().__init__(backbone, pool_type, checkpoint_batch_size, which_layer, lora_config, self_extend)

    def partial_encode(self, *inputs):
        input_embeddings, attention_mask = inputs
        backbone_outputs = self.backbone(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        backbone_full_embedding = backbone_outputs['hidden_states'][self.which_layer]

        return backbone_full_embedding

    def backbone_embedding(self, input_ids):    # embedding layer
        return self.backbone.embed_tokens(input_ids)

    def backbone_forward(self, input_ids, attention_mask):
        backbone_outputs = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        backbone_full_embedding = backbone_outputs['hidden_states'][self.which_layer]

        return backbone_full_embedding
    
class PhiEmbedder(BaseEmbedder):
    def __init__(self, backbone: str, PhiBackboneWrapper: BaseBackboneWrapper=PhiBackboneWrapper, pool_type: str='cls', checkpoint_batch_size=-1, embed_dim: int=-1, which_layer: int=-1, lora_config: bool=False, mytryoshka_indexes: list=None):
        super().__init__(backbone, PhiBackboneWrapper, pool_type, checkpoint_batch_size, embed_dim, which_layer, lora_config, mytryoshka_indexes)