import os
import torch
from transformers import AutoModel, AutoTokenizer
from .base_model import BaseBackboneWrapper, BaseEmbedder

class BGEBackboneWrapper(BaseBackboneWrapper):
    def __init__(self, backbone, pool_type: str = 'cls', checkpoint_batch_size: int = -1, which_layer: int = -1, lora_config: bool = True, self_extend: bool = False):
        # backbone = AutoModel.from_pretrained(backbone)
        super().__init__(backbone, pool_type, checkpoint_batch_size, which_layer, lora_config, self_extend)

    def partial_encode(self, *inputs):
        return super().partial_encode(*inputs)

    def backbone_embedding(self, input_ids):
        return super().backbone_embedding(input_ids)

    def backbone_forward(self, input_items):
        return super().backbone_forward(input_items)
    

class BGECustomEmbedder(BaseEmbedder):
    def __init__(self, backbone: str, backbone_wrapper: BaseBackboneWrapper=BGEBackboneWrapper, pool_type: str = 'cls', checkpoint_batch_size=-1, embed_dim: int = -1, which_layer: int = -1, lora_config: bool = False, mytryoshka_indexes: list = None, normalize: bool = False):
        super().__init__(backbone, backbone_wrapper, pool_type, checkpoint_batch_size, embed_dim, which_layer, lora_config, mytryoshka_indexes, normalize)

class BGEEmbedder():
    def __init__(self, backbone: str, device: str, max_length: int, ckpt: str=None) -> None:
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        model = AutoModel.from_pretrained(backbone)
        if ckpt:
            if os.path.exists(ckpt):
                print(f'>>> Loading BGEEmbedder CKPT from {ckpt}')
                model.load_state_dict(torch.load(ckpt))

        model.eval()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
        # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
        # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
        for k in encoded_input.keys():
            encoded_input[k] = encoded_input[k].to(self.model.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.detach().cpu()