import os
import math
import random
from typing import List
# from embedding.data.data_utils import dataset_sampling_ratios

dataset_sampling_ratios = {
    'ELI5': 0.1,
    'HotpotQA': 1.0,
    'MSMARCO': 0.5,
    'MultiNLI': 1.0,
    'Quora': 0.1,
    'MIRACL': 1.0,
    'MrTyDi': 1.0,
    'SQuAD': 1.0,
    'NautralQuestions': 1.0,
    'TriviaQA': 1.0,
    'FEVER': 1.0,
    'DuReader': 1.0,
    'T2Ranking': 1.0
}

random.seed(20)

def indomain_batch_merge(datafiles: List[str], batchsize: int, indomain_batch_dir: str):
    if not os.path.exists(indomain_batch_dir):
        os.makedirs(indomain_batch_dir)

    datasets = dict()
    for f in datafiles:
        dname = f.split('/')[-2]
        flines = [l for l in open(f, 'r').readlines()]
        chunk_cnt = math.ceil(len(flines) / batchsize)
        for c in range(chunk_cnt):
            if dname not in datasets:
                datasets[dname] = []
            
            datasets[dname].append(flines[c*batchsize: (c+1)*batchsize])

            if (c+1)*batchsize >= len(flines)*dataset_sampling_ratios[dname]:
                if len(datasets[dname][-1]) != batchsize:
                    del datasets[dname][-1]
                break

    all_batch_chunks = []
    for chunks in datasets.values():
        all_batch_chunks.extend(chunks)

    random.shuffle(all_batch_chunks)

    with open(os.path.join(indomain_batch_dir, 'train.jsonl'), 'w') as fw:
        for chunks in all_batch_chunks:
            for l in chunks:
                if not l.endswith('\n'):
                    l = l + '\n'
                
                fw.write(l)


if __name__ == '__main__':
    datafiles = [
        '/fs-computility/llm/chenzhi/datasets_processed/ELI5/train.jsonl', 
        '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/Quora/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train.jsonl'
    ]
    batchsize = 256
    indomain_batch_dir = '/fs-computility/llm/chenzhi/datasets_processed/InDomain'

    indomain_batch_merge(datafiles, batchsize, indomain_batch_dir)