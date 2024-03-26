import torch
from typing import Union
from peft import LoraConfig
from peft import get_peft_model, LoraConfig, TaskType
from .base_model import BaseBackboneWrapper, BaseEmbedder


class InternLMBackboneWrapper(BaseBackboneWrapper):
    def __init__(self, backbone: str, pool_type: str='cls', checkpoint_batch_size: int=-1, which_layer: int=-1, lora_config: Union[bool, LoraConfig]=True, self_extend: bool=False):
        super().__init__(backbone, pool_type, checkpoint_batch_size, which_layer, lora_config, self_extend)

    def partial_encode(self, *inputs):
        input_embeddings, attention_mask = inputs
        backbone_outputs = self.backbone(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        backbone_full_embedding = backbone_outputs['hidden_states'][self.which_layer]

        return backbone_full_embedding

    def backbone_embedding(self, input_ids):
        if self.lora_config:
            return self.backbone.model.model.tok_embeddings(input_ids)
        else:
            return self.backbone.model.tok_embeddings(input_ids)

    def backbone_forward(self, input_items):
        backbone_outputs = self.backbone(**input_items, output_hidden_states=True, return_dict=True)
        backbone_full_embedding = backbone_outputs['hidden_states'][self.which_layer]

        return backbone_full_embedding
    
    def lora_wrapper(self, model):
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, 
                                r=8, 
                                lora_alpha=32, # 32
                                lora_dropout=0.05,
                                target_modules=['wqkv', 'wo'], # None
                                bias='none')
        model = get_peft_model(model, peft_config)
        return model
    
class InternLMEmbedder(BaseEmbedder):
    def __init__(self, backbone: str, InternLMBackboneWrapper: BaseBackboneWrapper=InternLMBackboneWrapper, pool_type: str='cls', checkpoint_batch_size=-1, embed_dim: int=-1, which_layer: int=-1, lora_config: bool=True, mytryoshka_indexes: list=None, normalize: bool = False):
        super().__init__(backbone, InternLMBackboneWrapper, pool_type, checkpoint_batch_size, embed_dim, which_layer, lora_config, mytryoshka_indexes, normalize)