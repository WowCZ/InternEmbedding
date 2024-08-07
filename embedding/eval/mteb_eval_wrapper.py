import os
import math
import torch
from tqdm import tqdm
from mteb import MTEB
from tqdm.rich import trange
from typing import List, Union
from functools import partial
from transformers import AutoTokenizer
import torch.nn.functional as F
from embedding.models.base_model import BaseEmbedder
from embedding.data.data_loader import make_text_batch
from embedding.data.datasets import custom_corpus_prompt
from embedding.eval.eval_utils import get_task_def_by_task_name_and_type

class EvaluatedEmbedder:
    def __init__(self, embedder: Union[str, BaseEmbedder], tokenizer: AutoTokenizer, max_length: int, device: str='cuda:0', eval_batch_size: int=916):
        if type(embedder) is str:
            self.embedder = torch.load(embedder)
        else:
            self.embedder = embedder
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.eval_batch_size = eval_batch_size

    def batch_encode(self, sentences: Union[str, list], batch_size=32, **kwargs):
        batch_size = self.eval_batch_size
        if type(sentences) is str:
            sentences = [sentences]

        batch_cnt = math.ceil(len(sentences) / batch_size)
        sentence_embeddings = []
        with torch.no_grad():
            for bi in trange(batch_cnt):
                cur_batch = sentences[bi*batch_size: (bi+1)* batch_size]
                bi_inputs = make_text_batch(cur_batch, self.tokenizer, self.max_length, self.device)

                cur_embeddings = self.embedder.embedding(bi_inputs)
                sentence_embeddings.append(cur_embeddings)

        return torch.cat(sentence_embeddings, dim=0).detach().cpu()
    
    def batch_encode_similarity(self, sentences: Union[str, list], batch_size=32, **kwargs):
        # batch_size = self.eval_batch_size
        if type(sentences) is str:
            sentences = [sentences]

        batch_cnt = math.ceil(len(sentences) / batch_size)
        sentence_embeddings = []
        with torch.no_grad():
            for bi in trange(batch_cnt):
            # for bi in range(batch_cnt):
                cur_batch = sentences[bi*batch_size: (bi+1)* batch_size]
                bi_inputs = make_text_batch(cur_batch, self.tokenizer, self.max_length, self.device)

                cur_embeddings = self.embedder.embedding(bi_inputs)
                sentence_embeddings.append(cur_embeddings)

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        similarity = sentence_embeddings.mm(sentence_embeddings.t())
        return sentence_embeddings.detach().cpu(), similarity.detach().cpu()
    
    def encode_queries(self, queries, prompt=None, batch_size=32, **kwargs):
        if prompt is not None:
            queries = [prompt + ': ' + s for s in queries]

        return self.batch_encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus, prompt=None, task_type=None, batch_size=32, **kwargs):
        if type(corpus[0]) is dict:
            corpus = [
                (doc["title"] + '\n' + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]

        if prompt is not None:
            prompt = custom_corpus_prompt(prompt, task_type)
            corpus = [prompt + ': ' + s for s in corpus]

        return self.batch_encode(corpus, batch_size=batch_size, **kwargs)
    
    def encode(self, sentences: list, prompt: str=None, batch_size=32, **kwargs):
        if prompt is not None:
            sentences = [prompt + ': ' + s for s in sentences]

        return self.batch_encode(sentences, batch_size=batch_size, **kwargs)

class MTEBEvaluationWrapper:
    def __init__(self, embedder: Union[str, BaseEmbedder], model_name: str, tokenizer: AutoTokenizer, max_length: int, prompt: bool, device: str, result_dir: str):
        self.evaluated_embedder = EvaluatedEmbedder(embedder, tokenizer, max_length, device)
        self.model_name = model_name
        self.prompt = prompt
        self.result_dir = result_dir

    def evaluation(self, mteb_tasks: List[str]):
        results = []
        for task in mteb_tasks:
            evaluation = MTEB(tasks=[task], task_langs=['en'])
            task_cls = evaluation.tasks[0]
            task_name = task_cls.description['name']
            task_type = task_cls.description['type']

            if self.prompt:
                prompt = get_task_def_by_task_name_and_type(task_name=task_name, task_type=task_type)
                if task_type in ['Retrieval', 'Reranking']:
                    self.evaluated_embedder.encode_queries = partial(self.evaluated_embedder.encode_queries, prompt=prompt)
                    self.evaluated_embedder.encode_corpus = partial(self.evaluated_embedder.encode_corpus, prompt=prompt, task_type=task_type)
                else:
                    self.evaluated_embedder.encode = partial(self.evaluated_embedder.encode, prompt=prompt)

            result = evaluation.run(self.evaluated_embedder, output_folder=os.path.join(self.result_dir, self.model_name))
            results.append(result)

        return results
