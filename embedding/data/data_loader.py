import torch
from functools import partial
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

def make_text_batch(t_ls: list, tokenizer: AutoTokenizer, max_length: int, device: str=None):
    if any(x is None for x in t_ls):
        return None, None
    
    tokens = tokenizer(t_ls, padding='max_length', max_length=max_length, return_tensors='pt', truncation=True)
    if device:
        for k in tokens.keys():
            tokens[k] = tokens[k].to(device)
    return tokens

def make_query_passage_batch(qp_ls: list, tokenizer: AutoTokenizer, max_length: int):
    q_ls = [qp[0] for qp in qp_ls]
    p_ls = [qp[1] for qp in qp_ls]
    n_ls = [qp[2] for qp in qp_ls]

    q_inputs = make_text_batch(q_ls, tokenizer, max_length)
    p_inputs = make_text_batch(p_ls, tokenizer, max_length)

    if any(x is None for x in n_ls):
        n_list_inputs = None
    else:
        n_list_inputs = []
        negative_cnt = len(n_ls[0])
        for nid in range(negative_cnt):
            n_inputs = make_text_batch([ns[nid] for ns in n_ls], tokenizer, max_length)
            n_list_inputs.append(n_inputs)

    return q_inputs, p_inputs, n_list_inputs


def train_dataloader(qp_pairs: Dataset, tokenizer: AutoTokenizer, max_length: int, sampler: str, batch_size: int):
    if sampler == 'random':
        train_sampler = RandomSampler(qp_pairs)
    elif sampler == 'sequential':
        train_sampler = SequentialSampler(qp_pairs)

    collate_fn = partial(make_query_passage_batch, tokenizer=tokenizer, max_length=max_length)

    train_batch_loader = DataLoader(
        qp_pairs,
        collate_fn=collate_fn,
        batch_size=batch_size,
        sampler=train_sampler
    )

    return train_batch_loader