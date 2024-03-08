import torch
from functools import partial
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

def make_text_batch(t_ls: list, tokenizer: AutoTokenizer, max_length: int):
    if any(x is None for x in t_ls):
        return None, None
    
    tokens = tokenizer(t_ls, padding='max_length', max_length=max_length, truncation=True)
    inputs, attention_mask = torch.LongTensor(tokens['input_ids']), torch.LongTensor(tokens['attention_mask'])
    return inputs, attention_mask

def make_query_passage_batch(qp_ls: list, tokenizer: AutoTokenizer, max_length: int):
    q_ls = [qp[0] for qp in qp_ls]
    p_ls = [qp[1] for qp in qp_ls]
    n_ls = [qp[2] for qp in qp_ls]

    q_inputs, q_attention_mask = make_text_batch(q_ls, tokenizer, max_length)
    p_inputs, p_attention_mask = make_text_batch(p_ls, tokenizer, max_length)

    if any(x is None for x in n_ls):
        n_list_inputs, n_list_attention_mask = None, None
    else:
        n_list_inputs, n_list_attention_mask = [], []
        negative_cnt = len(n_ls[0])
        for nid in range(negative_cnt):
            n_inputs, n_attention_mask = make_text_batch([ns[nid] for ns in n_ls], tokenizer, max_length)
            n_list_inputs.append(n_inputs)
            n_list_attention_mask.append(n_attention_mask)

    return q_inputs, q_attention_mask, p_inputs, p_attention_mask, n_list_inputs, n_list_attention_mask


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