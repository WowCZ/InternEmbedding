import os
import math
import tqdm
import torch
from embedding.models.modeling_bge import BGECustomEmbedder
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoTokenizer
from embedding.data.data_loader import make_text_batch
import torch.nn.functional as F


class BGEFunction(EmbeddingFunction):
    def __init__(self, bge_name, bge_ckpt: str, *args, **kwargs):
        super(BGEFunction, self).__init__(*args, **kwargs)
        self.bge_name = bge_name
        print(f'>>> Loading BGE embedder from {self.bge_name}')
        # self.embedder = BGEEmbedder(self.bge_name, device='cuda', max_length=512, ckpt=ckpt)
        self.device = 'cuda'
        self.embedder = BGECustomEmbedder(bge_name, pool_type='cls', checkpoint_batch_size=-1, embed_dim=-1, lora_config=False, which_layer=-1, mytryoshka_indexes=None, normalize=True).to(self.device)
        if bge_ckpt:
            if os.path.exists(bge_ckpt):
                print(f'>>> Loading BGEEmbedder CKPT from {bge_ckpt}')
                self.embedder.load_state_dict(torch.load(bge_ckpt))
        
        self.embedder.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(bge_name)
        self.max_length = 512

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents
        doc_cnt = len(input)
        batch_size = 128 * 15
        batch_num = math.ceil(doc_cnt / batch_size)

        doc_embeddings = []
        for bi in tqdm.tqdm(range(batch_num)):
            cur_batch = input[bi*batch_size : (bi+1)*batch_size]
            bi_inputs = make_text_batch(cur_batch, self.tokenizer, self.max_length, self.device)
            
            with torch.no_grad():
                cur_embeddings = self.embedder.embedding(bi_inputs)

            doc_embeddings.append(cur_embeddings)

        return torch.cat(doc_embeddings, dim=0).detach().cpu().numpy().tolist()