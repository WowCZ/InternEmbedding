import torch
from typing import Union, List
from peft import LoraConfig
from peft import get_peft_model, LoraConfig, TaskType
from .base_model import BaseBackboneWrapper, BaseEmbedder
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM


class InternLMBackboneWrapper(BaseBackboneWrapper):
    def __init__(self, 
                 backbone: str, 
                 pool_type: str='cls', 
                 checkpoint_batch_size: int=-1, 
                 which_layer: int=-1, 
                 reserved_layers: List[int]=None,
                 lora_config: Union[bool, LoraConfig]=True, 
                 self_extend: bool=False,
                 **kwargs):
        super().__init__(backbone, 
                         pool_type, 
                         checkpoint_batch_size, 
                         which_layer, 
                         reserved_layers, 
                         lora_config, 
                         self_extend,
                         **kwargs)

    def _init_backbone(self, backbone, **kwargs):
        flashatt = kwargs['flashatt']
        config = AutoConfig.from_pretrained(backbone, trust_remote_code=True)

        if flashatt:
            config._flash_attn_2_enabled = True
            config.attn_implementation = "flash_attention_2"

        try:
            backbone = AutoModelForCausalLM.from_pretrained(backbone, 
                                                            config=config, 
                                                            trust_remote_code=True)  # , local_files_only=True: PretrainedConfig
        except:
            backbone = AutoModel.from_pretrained(backbone, 
                                                 config=config, 
                                                 trust_remote_code=True)
                
        return backbone

    def partial_encode(self, *inputs):
        input_embeddings, attention_mask = inputs
        backbone_outputs = self.backbone(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        backbone_full_embedding = backbone_outputs['hidden_states'][-1]

        return backbone_full_embedding

    def backbone_embedding(self, input_ids):
        if self.lora_config:
            return self.backbone.model.model.tok_embeddings(input_ids)
        else:
            return self.backbone.model.tok_embeddings(input_ids)

    def backbone_forward(self, input_items):
        backbone_outputs = self.backbone(**input_items, output_hidden_states=True, return_dict=True)
        backbone_full_embedding = backbone_outputs['hidden_states'][-1]

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
    
    def model_razor(self, backbone, reserved_layers):
        for lay in range(backbone.config.num_hidden_layers)[::-1]:
            if lay not in reserved_layers:
                del(backbone.model.layers[lay])
        # reset config
        backbone.config.num_hidden_layers = len(backbone.model.layers)
        print('Current model layers: ', backbone.config.num_hidden_layers)              
        return backbone
    
class InternLMEmbedder(BaseEmbedder):
    def __init__(self, 
                 backbone: str, 
                 InternLMBackboneWrapper: BaseBackboneWrapper=InternLMBackboneWrapper, 
                 pool_type: str='cls', 
                 checkpoint_batch_size=-1, 
                 flashatt: bool=False,
                 which_layer: int=-1, 
                 reserved_layers: List[int]=None,
                 lora_config: bool=True, 
                 mytryoshka_indexes: list=None, 
                 normalize: bool = False,
                 **kwargs):
        super().__init__(backbone, 
                         InternLMBackboneWrapper, 
                         pool_type, 
                         checkpoint_batch_size, 
                         flashatt, 
                         which_layer, 
                         reserved_layers, 
                         lora_config, 
                         mytryoshka_indexes, 
                         normalize, 
                         **kwargs)