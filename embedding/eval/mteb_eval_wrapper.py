import math
from tqdm.rich import trange
import torch
from mteb import MTEB
from typing import List, Union
from functools import partial
from transformers import AutoTokenizer
import torch.nn.functional as F
from embedding.models.base_model import BaseEmbedder
from embedding.data.data_loader import make_text_batch
from embedding.eval.eval_utils import get_task_def_by_task_name_and_type

class EvaluatedEmbedder:
    def __init__(self, embedder: Union[str, BaseEmbedder], tokenizer: AutoTokenizer, max_length: int, embedding_norm: bool=True, device: str='cuda:0'):
        if type(embedder) is str:
            self.embedder = torch.load(embedder)
        else:
            self.embedder = embedder
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embedding_norm = embedding_norm
        self.device = device

    def encode(self, sentences: Union[str, list], batch_size=32, prompt=None, **kwargs):
        if type(sentences) is str:
            sentences = [sentences]
            
        if prompt: 
            sentences = [prompt + ': ' + s for s in sentences]
            # print('>>> prompt: ' + prompt)

        batch_cnt = math.ceil(len(sentences) / batch_size)
        self.embedder.eval()
        sentence_embeddings = []
        with torch.no_grad():
            for bi in trange(batch_cnt):
            # for bi in range(batch_cnt):
                cur_batch = sentences[bi*batch_size: (bi+1)* batch_size]
                bi_input_ids, bi_attention_mask = make_text_batch(cur_batch, self.tokenizer, self.max_length)
                bi_input_ids, bi_attention_mask = bi_input_ids.to(self.device), bi_attention_mask.to(self.device)
                cur_embeddings = self.embedder.embedding(bi_input_ids, bi_attention_mask)
                if self.embedding_norm:
                    cur_embeddings = F.normalize(cur_embeddings, p=2, dim=-1)
                sentence_embeddings.append(cur_embeddings)

        return torch.cat(sentence_embeddings, dim=0).detach().cpu()


class MTEBEvaluationWrapper:
    def __init__(self, embedder: Union[str, BaseEmbedder], model_name: str, tokenizer: AutoTokenizer, max_length: int, prompt: bool, embedding_norm: bool, device: str):
        self.evaluated_embedder = EvaluatedEmbedder(embedder, tokenizer, max_length, embedding_norm, device)
        self.model_name = model_name
        self.prompt = prompt

    def evaluation(self, mteb_tasks: List[str]):
        results = []
        for task in mteb_tasks:
            evaluation = MTEB(tasks=[task], task_langs=['en'])
            task_cls = evaluation.tasks[0]
            task_name = task_cls.description['name']
            task_type = task_cls.description['type']

            if self.prompt:
                prompt = get_task_def_by_task_name_and_type(task_name=task_name, task_type=task_type)
                self.evaluated_embedder.encode = partial(self.evaluated_embedder.encode, prompt=prompt)
            result = evaluation.run(self.evaluated_embedder, output_folder=f"results/{self.model_name}")
            results.append(result)

        return results
