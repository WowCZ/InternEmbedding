import math
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoConfig
import torch.utils.checkpoint as checkpoint
from embedding.eval.mistral_sel_extend_patch import self_extend_forward, modify_method_of_instance

class BaseBackboneWrapper(nn.Module, ABC):
    def __init__(self, backbone, pool_type: str='position_weight', checkpoint_batch_size: int=-1, which_layer: int=-1, lora_config: bool=True, self_extend: bool=False):
        super(BaseBackboneWrapper, self).__init__()
        # initial backbone model
        if self_extend:
            mistral_self_extend_forward = partial(self_extend_forward, group_size_1=4, group_size_2=512)
            config = AutoConfig.from_pretrained(backbone)
            config.sliding_window = 200000000
            backbone = AutoModel.from_pretrained(backbone, config=config)
            modify_method_of_instance(backbone, "MistralAttention", "forward", mistral_self_extend_forward)
        else:
            backbone = AutoModel.from_pretrained(backbone, trust_remote_code=True)

        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            backbone.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # backbone.gradient_checkpointing_enable()

        if lora_config:
            backbone = self.lora_wrapper(backbone)
            # backbone.print_trainable_parameters()

        backbone.config.use_cache = False
        self.backbone = backbone
        if pool_type not in ['cls', 'eos', 'position_weight', 'mean']:
            raise TypeError(f"There is an unidentified pooling type {self.pool_type}.")
        
        self.pool_type = pool_type
        self.checkpoint_batch_size = checkpoint_batch_size
        self.which_layer = which_layer
        self.lora_config = lora_config

    @abstractmethod
    def lora_wrapper(self, model):
        return model

    @abstractmethod
    def partial_encode(self, *inputs):
        # directly input the representation of the embedding layer for checkpointing operation
        input_embeddings, attention_masks = inputs
        extended_attention_mask = self.backbone.get_extended_attention_mask(attention_masks, input_embeddings.size()[:2])
        backbone_outputs = self.backbone.encoder(input_embeddings, attention_mask=extended_attention_mask)[0]
        return backbone_outputs

    @abstractmethod
    def backbone_embedding(self, input_ids: torch.LongTensor):
        return self.backbone.embeddings(input_ids)

    @abstractmethod
    def backbone_forward(self, input_items):
        # return the hidden state of the backbone
        return self.backbone(**input_items)[0]

    def grad_cache_checkpoint(self, input_items):
        # if input batch size is much larger then checkpoint_batch_size, we need to use checkpoint to aggragate the gradient cache.
        if self.checkpoint_batch_size == -1 or input_items['input_ids'].size(0) < self.checkpoint_batch_size:
            return self.backbone_forward(input_items)
        else:
            backbone_output_list = []
            cache_nums = math.ceil(input_items['input_ids'].size(0) / self.checkpoint_batch_size)
            for step in range(cache_nums):
                left, right = step*self.checkpoint_batch_size, (step+1)*self.checkpoint_batch_size
                cur_input_ids = input_items['input_ids'][left: right]
                cur_input_embeddings = self.backbone_embedding(cur_input_ids)
                cur_attention_masks = input_items['attention_mask'][left: right]
                cur_backbone_outputs = checkpoint.checkpoint(self.partial_encode, cur_input_embeddings, cur_attention_masks, use_reentrant=False)
                backbone_output_list.append(cur_backbone_outputs)
            return torch.cat(backbone_output_list, dim=0)

    def encode(self, input_items):
        # return the presentation vectors at the last layer of the backbone
        cached_backbone_outputs = self.grad_cache_checkpoint(input_items)

        # pooling operation
        batch_size = input_items['attention_mask'].size(0)
        # identify the padding strategy in the batch (left padding or right padding)
        left_padding = (input_items['attention_mask'][:, -1].sum() == batch_size)
        input_lens = input_items['attention_mask'].sum(dim=1)
        if self.pool_type == 'cls':
            if not left_padding:
                return cached_backbone_outputs[:, 0]
            else:
                return cached_backbone_outputs[torch.arange(batch_size, device=cached_backbone_outputs.device), batch_size-input_lens]
        elif self.pool_type == 'eos':
            if not left_padding:
                return cached_backbone_outputs[torch.arange(batch_size, device=cached_backbone_outputs.device), input_lens-1]
            else:
                return cached_backbone_outputs[:, -1]
        elif self.pool_type == 'position_weight':
            position_weight = input_items['attention_mask'].cumsum(dim=-1)
            position_weight = position_weight / position_weight.sum(dim=1).unsqueeze(1)
            return (cached_backbone_outputs * position_weight.unsqueeze(2)).sum(dim=1)
        elif self.pool_type == 'mean':
            mean_weight = input_items['attention_mask'] / input_items['attention_mask'].sum(dim=1).unsqueeze(1)
            return (cached_backbone_outputs * mean_weight.unsqueeze(2)).sum(dim=1)
        
        raise TypeError(f"There is an unidentified pooling type {self.pool_type}.")
    
    @property
    def backbone_dim(self):
        try:
            tmp_ids = torch.LongTensor([[0]])
            tmp_type = torch.LongTensor([[0]])
            tmp_msk = torch.LongTensor([[1]])
            tmp_item = {
                'input_ids': tmp_ids,
                'attention_mask': tmp_msk,
                'token_type_ids': tmp_type
            }
            backbone_dim = self.backbone_forward(tmp_item).size(-1)
        except:
            tmp_ids = torch.LongTensor([[0]])
            tmp_msk = torch.LongTensor([[1]])
            tmp_item = {
                'input_ids': tmp_ids,
                'attention_mask': tmp_msk
            }
            backbone_dim = self.backbone_forward(tmp_item).size(-1)
        return backbone_dim


class BaseEmbedder(nn.Module, ABC):
    def __init__(self, backbone: str, backbone_wrapper: BaseBackboneWrapper, pool_type: str = 'cls', checkpoint_batch_size=-1, embed_dim: int=-1, which_layer: int=-1, lora_config: bool=True, mytryoshka_indexes: list=None, normalize: bool=False):
        super(BaseEmbedder, self).__init__()

        self.encoder = backbone_wrapper(backbone, pool_type, checkpoint_batch_size, which_layer, lora_config)
        # self.device = self.encoder.backbone.device

        if embed_dim == -1:
            self.embed_dim = self.encoder.backbone_dim
            self.project = None
        else:
            self.embed_dim = embed_dim
            self.project = nn.Linear(self.encoder.backbone_dim, embed_dim, bias=False)
        # print(f'>>> The dimension of {backbone} is {self.embed_dim}.')
        self.which_layer = which_layer
        self.mytryoshka_indexes = mytryoshka_indexes
        self.normalize = normalize

    def embedding(self, input_items):
        embeddings = self.encoder.encode(input_items)
        if self.project is not None:
            embeddings = self.project(embeddings)

        device = embeddings.device
        if self.mytryoshka_indexes:
            mytryoshka_embedding = embeddings.index_select(-1, torch.tensor(self.mytryoshka_indexes).to(device))
        else:
            mytryoshka_embedding = embeddings

        if self.normalize:
            mytryoshka_embedding = F.normalize(mytryoshka_embedding, p=2, dim=-1)
        return mytryoshka_embedding
    
    def forward(self, q_ids, p_ids, n_list_ids):
        q_embeddings = self.embedding(q_ids)
        p_embeddings = self.embedding(p_ids)
        n_m_embeddings = None
        if n_list_ids is not None:
            if type(n_list_ids) is not list:
                n_list_ids = [n_list_ids]
            
            n_m_embeddings = []
            for n_ids in n_list_ids:
                n_embeddings = self.embedding(n_ids)
                n_m_embeddings.append(n_embeddings.unsqueeze(1))

        return q_embeddings, p_embeddings, torch.cat(n_m_embeddings, dim=1) if n_m_embeddings is not None else None