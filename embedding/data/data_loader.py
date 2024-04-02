import torch
from functools import partial
from transformers import AutoTokenizer
from embedding.data.data_utils import InDatasetSampler, InDatasetBatchSampler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, BatchSampler


def expand_list(nested_list):
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            for sub_item in expand_list(item):
                yield sub_item
        else:
            yield item

def make_text_batch(t_ls: list, tokenizer: AutoTokenizer, max_length: int, device: str=None):
    if any(x is None for x in t_ls):
        return None, None
    
    tokens = tokenizer(t_ls, 
                       padding='max_length', 
                       max_length=max_length, 
                       return_tensors='pt', 
                       truncation=True)
    if device is not None:
        for k in tokens.keys():
            tokens[k] = tokens[k].to(device)
    return tokens

def make_query_passage_batch(qp_ls: list, tokenizer: AutoTokenizer, max_length: int):
    q_ls = [qp[0] for qp in qp_ls]
    p_ls = [qp[1] for qp in qp_ls]
    n_ls = [qp[2] for qp in qp_ls]
    t_ls = [qp[3] for qp in qp_ls] # Task types

    # task_type_inbatch = 'DIVERSITY'
    # if all(t == t_ls[0] for t in t_ls):
    #     task_type_inbatch = t_ls[0]

    q_inputs = make_text_batch(q_ls, tokenizer, max_length)
    p_inputs = make_text_batch(p_ls, tokenizer, max_length)

    if any(x is None for x in list(expand_list(n_ls))):
        n_list_inputs = []
    else:
        n_list_inputs = []
        negative_cnt = min([len(ns) for ns in n_ls])
        for nid in range(negative_cnt):
            n_inputs = make_text_batch([ns[nid] for ns in n_ls], tokenizer, max_length)
            n_list_inputs.append(n_inputs)

    return (q_inputs, p_inputs, n_list_inputs, t_ls)


def train_dataloader(qp_pairs: Dataset, tokenizer: AutoTokenizer, max_length: int, sampler: str, batch_size: int):
    if sampler == 'random':
        train_sampler = RandomSampler(qp_pairs)
        batch_sampler = BatchSampler(train_sampler, batch_size=batch_size, drop_last=False)
    elif sampler == 'sequential':
        train_sampler = SequentialSampler(qp_pairs)
        batch_sampler = BatchSampler(train_sampler, batch_size=batch_size, drop_last=False)
    elif sampler == 'indataset':
        train_sampler = InDatasetSampler(qp_pairs, batch_size=batch_size)
        batch_sampler = InDatasetBatchSampler(train_sampler, batch_size=batch_size, drop_last=False)

    collate_fn = partial(make_query_passage_batch, tokenizer=tokenizer, max_length=max_length)

    train_batch_loader = DataLoader(
        qp_pairs,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler
    )

    return train_batch_loader