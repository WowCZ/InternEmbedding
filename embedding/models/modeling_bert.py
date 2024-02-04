from .base_model import BaseBackboneWrapper, BaseEmbedder
from transformers import AutoModel, AutoTokenizer

class BertBackboneWrapper(BaseBackboneWrapper):
    def __init__(self, backbone: str, pool_type: str = 'cls', checkpoint_batch_size=-1, device='cuda:0'):
        backbone = AutoModel.from_pretrained(backbone)
        super().__init__(backbone, pool_type, checkpoint_batch_size)

    def partial_encode(self, *inputs):
        return super().partial_encode(*inputs)

    def backbone_embedding(self, input_ids):
        return super().backbone_embedding(input_ids)

    def backbone_forward(self, input_ids, attention_mask):
        return super().backbone_forward(input_ids, attention_mask)
    
class BertEmbedder(BaseEmbedder):
    def __init__(self, backbone: str, BertBackboneWrapper: BaseBackboneWrapper = BertBackboneWrapper, pool_type: str = 'cls', checkpoint_batch_size=-1, embed_dim: int = 1024, device: str='cuda'):
        super().__init__(backbone, BertBackboneWrapper, pool_type, checkpoint_batch_size, embed_dim, device)