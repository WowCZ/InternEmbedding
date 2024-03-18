import os
import math
import torch
from mteb import MTEB
from tqdm.rich import trange
from typing import List, Union
from functools import partial
from transformers import AutoTokenizer
import torch.nn.functional as F
from embedding.models.base_model import BaseEmbedder
from embedding.data.data_loader import make_text_batch
from embedding.eval.eval_utils import get_task_def_by_task_name_and_type

class EvaluatedEmbedder:
    def __init__(self, embedder: Union[str, BaseEmbedder], tokenizer: AutoTokenizer, max_length: int, device: str='cuda:0', eval_batch_size: int=64):
        if type(embedder) is str:
            self.embedder = torch.load(embedder)
        else:
            self.embedder = embedder
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.eval_batch_size = eval_batch_size

    def encode(self, sentences: Union[str, list], prompt=None, batch_size=32, **kwargs):
        # batch_size = self.eval_batch_size
        if type(sentences) is str:
            sentences = [sentences]
            
        if prompt: 
            sentences = [prompt + ': ' + s for s in sentences]
            # print('>>> prompt: ' + prompt)

        batch_cnt = math.ceil(len(sentences) / batch_size)
        sentence_embeddings = []
        with torch.no_grad():
            for bi in trange(batch_cnt):
            # for bi in range(batch_cnt):
                cur_batch = sentences[bi*batch_size: (bi+1)* batch_size]
                bi_inputs = make_text_batch(cur_batch, self.tokenizer, self.max_length, self.device)

                cur_embeddings = self.embedder.embedding(bi_inputs)
                sentence_embeddings.append(cur_embeddings)

        return torch.cat(sentence_embeddings, dim=0).detach().cpu()


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
                self.evaluated_embedder.encode = partial(self.evaluated_embedder.encode, prompt=prompt)
            result = evaluation.run(self.evaluated_embedder, output_folder=os.path.join(self.result_dir, self.model_name))
            results.append(result)

        return results
