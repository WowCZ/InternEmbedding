import os
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from embedding.eval.metrics import cosine_similarity
from embedding.train.training_embedder import initial_model
from embedding.eval.mteb_eval_wrapper import EvaluatedEmbedder
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def extract_data_dicts(file_path):
    sample_list = []
    extracted = []
    with open(file_path, 'r') as f:
        for line in f:
            sample_list.append(json.loads(line))
    for sample in sample_list:
        for text_dict in sample['texts']['texts']:
            extracted.append(text_dict.copy())
    return extracted


def predict_embedder(args):
    embedder, tokenizer = initial_model(args)
    embedder.load_state_dict(torch.load(args.embedder_ckpt_path))
    embedder.eval()
    embedder = embedder.to(args.device)

    evaluated_embedder = EvaluatedEmbedder(embedder, tokenizer, args.max_length, args.device)
    sample_list = extract_data_dicts(args.dataset_path)

    os.makedirs(args.result_dir, exist_ok=True)
    result_file = Path(args.result_dir) / 'score_logits.jsonl'
    with open(result_file, 'w') as f:
        pass
    for sample in tqdm(sample_list, ncols=66):
        text = sample['content']
        logit = evaluated_embedder.encode(text)
        sample['ad_logit_0326'] = float(logit.view(-1).cpu())
        with open(result_file, 'a') as f:
            f.write(json.dumps(sample) + '\n')
