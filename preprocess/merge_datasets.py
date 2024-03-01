import os
import math
import json
import tqdm
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

dataset_task_prompts = {
    'classification': [
        'Given a premise, retrieve a hypothesis that is entailed by the premise',
        'Retrieve semantically similar text'
    ],
    'clustering': [
        'Identify the main category of the given text'
    ],
    'msmarco-triplets-shuffled': [
        'Given a web search query, retrieve relevant passages that answer the query',
        'Given a web search query, retrieve relevant documents that answer the query'
    ],
    'nq-triples-hardbatched': [
        'Given a question, retrieve Wikipedia passages that answer the question',
        'Retrieve Wikipedia passages that answer the question'
    ],
    'reddit_triples': [
        'Given a Reddit topic, retrieve relevant body that answer the Reddit'
    ],
    'retrieval': [
        'Given a question, retrieve the content that answer the question'
    ],
    'nli-triplets': [
        'Given a premise, retrieve a hypothesis that is entailed by the premise',
        'Retrieve semantically similar text'
    ],
    'specter_triples': [
        'Provided a title of the scientific publication, retrieve the related title of the publication'
    ]
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


def nomic_triple_aggregate(source_dir: str, target_dir: str):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    triples = []
    q_tags = []
    r_tags = []
    n_tags = []
    for root, _, fs in os.walk(source_dir):
        for f in tqdm.tqdm(fs):
            if f.endswith('.jsonl'):
                dataset = root.split('/')[-1]
                if dataset not in dataset_task_prompts:
                    print(dataset)
                    continue
                triple_f = os.path.join(root, f)
                for l in open(triple_f, 'r').readlines():
                    l = json.loads(l)
                    # if 'triplet' in l:
                    #     print(l['metadata']['triplet'])

                    prompt = dataset_task_prompts[dataset][0]
                    if len(l['metadata']['objective']['triplet']) == 0:
                        print(dataset)
                        break
                    if len(l['metadata']['objective']['triplet']) > 0:
                        q_tag = l['metadata']['objective']['triplet'][0][0]
                        r_tag = l['metadata']['objective']['triplet'][0][1]
                        n_tag = l['metadata']['objective']['triplet'][0][2]

                        if q_tag not in q_tags:
                            q_tags.append(q_tag)
                        if r_tag not in r_tags:
                            r_tags.append(r_tag)
                        if n_tag not in n_tags:
                            n_tags.append(n_tag)

                        triples.append({
                            'question': prompt + ' ' + l[q_tag],
                            'response': prompt + ' ' + l[r_tag],
                            'negative_response': prompt + ' ' + l[n_tag][0] if type(l[n_tag]) is list else prompt + ' ' + l[n_tag]
                        })

    print(q_tags, r_tags, n_tags)
    print('>>> Total Triples in Nomic: ', len(triples))
    with open(os.path.join(target_dir, 'train.jsonl'), 'w') as fw:
        for t in triples:
            fw.write(json.dumps(t) + '\n')


if __name__ == '__main__':
    # datafiles = [
    #     '/fs-computility/llm/chenzhi/datasets_processed/ELI5/train.jsonl', 
    #     '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/Quora/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train.jsonl',
    #     '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train.jsonl'
    # ]
    # batchsize = 256
    # indomain_batch_dir = '/fs-computility/llm/chenzhi/datasets_processed/InDomain'

    # indomain_batch_merge(datafiles, batchsize, indomain_batch_dir)

    source_dir = '/fs-computility/llm/shared/chenzhi/contrastive'
    target_dir = '/fs-computility/llm/chenzhi/datasets_processed/NOMICTriples'
    nomic_triple_aggregate(source_dir, target_dir)