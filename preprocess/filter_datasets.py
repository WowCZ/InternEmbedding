import json
import tqdm
import fasttext
import numpy as np
from typing import List

def cosine_similarity(x, y):
    # Ensure length of x and y are the same
    if len(x) != len(y) :
        return None
    
    # Compute the dot product between x and y
    dot_product = np.dot(x, y)
    
    # Compute the L2 norms (magnitudes) of x and y
    magnitude_x = np.sqrt(np.sum(x**2)) 
    magnitude_y = np.sqrt(np.sum(y**2))
    
    # Compute the cosine similarity
    cosine_similarity = dot_product / (magnitude_x * magnitude_y)
    
    return cosine_similarity

def language_filter(paired_data: List[str], source_langs: List[str]):
    filtered_data = []
    lang_cls_model = fasttext.load_model('/fs-computility/llm/chenzhi/fasttext_models/lid.176.bin')
    for l in tqdm.tqdm(paired_data):
        l = json.loads(l)
        q_lang, _ = lang_cls_model.predict(l['question'].split('\n')[0], k=1)
        q_lang = q_lang[0].split('__')[-1]
        if q_lang not in source_langs:
            continue

        filtered_data.append(json.dumps(l))

    return filtered_data

def deduplicate_filter(paired_data: List[str]):
    filtered_data = []
    hash_map = dict()
    for l in tqdm.tqdm(paired_data):
        l = json.loads(l)
        question = l['question']
        response = l['response']

        if question in hash_map or question.lower() == response.lower():
            continue

        hash_map[question] = 1.0
        filtered_data.append(json.dumps(l))

    return filtered_data

def consistency_filter(paired_data: List[str], embedder):
    filtered_data = []
    batch_size = 128
    batch_question, batch_response, batch_sample = [], [], []
    for li, l in tqdm.tqdm(enumerate(paired_data)):
        l = json.loads(l)
        question = l['question']
        response = l['response']
        batch_question.append(question)
        batch_response.append(response)
        batch_sample.append(l)

        if (li+1) % batch_size == 0:
            question_emb = embedder.encode(batch_question).numpy()
            response_emb = embedder.encode(batch_response).numpy()

            for bi, bl in enumerate(batch_sample):
                cosine_score = cosine_similarity(question_emb[bi], response_emb[bi])
                bl['cosine_score'] = str(cosine_score)
                filtered_data.append((cosine_score, json.dumps(bl)))

            batch_question, batch_response, batch_sample = [], [], []
    
    if len(batch_sample) > 0:
        question_emb = embedder.encode(batch_question).numpy()
        response_emb = embedder.encode(batch_response).numpy()

        for bi, bl in enumerate(batch_sample):
            cosine_score = cosine_similarity(question_emb[bi], response_emb[bi])
            bl['cosine_score'] = str(cosine_score)
            filtered_data.append((cosine_score, json.dumps(bl)))

    return sorted(filtered_data, key=lambda x:x[0], reverse=True)

if __name__ == '__main__':
    filter_phrase = 2

    if filter_phrase == 1:
        import os
        import sys
        import math
        import torch
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from embedding.eval.mteb_eval_wrapper import EvaluatedEmbedder
        from embedding.models.modeling_mistral import MistralEmbedder
        from embedding.models.modeling_bge import BGEEmbedder
        from transformers import AutoTokenizer

        mytryoshka_size = 4096
        max_length = 512
        embedding_norm = False
        mytryoshka_indexes = list(range(mytryoshka_size))
        rank = 7
        device = f'cuda:{rank}'
        print(f'>>> load in {device}!')

        # # for mistral embedder
        # backbone = "/fs-computility/llm/chenzhi/huggingface_cache/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/9ab9e76e2b09f9f29ea2d56aa5bd139e4445c59e"
        # tokenizer = AutoTokenizer.from_pretrained(backbone)
        # tokenizer.pad_token = tokenizer.eos_token
        # mistral_embedder = MistralEmbedder(backbone, pool_type='position_weight', checkpoint_batch_size=32, embed_dim=-1, lora_config=True, which_layer=-1, mytryoshka_indexes=mytryoshka_indexes).to(device)
        # mistral_embedder.load_state_dict(torch.load('/fs-computility/llm/chenzhi/ckpts/mistral_embedder13_paired_prompt_ml512_1261.pt'))
        # embedder = EvaluatedEmbedder(mistral_embedder, tokenizer, max_length, embedding_norm, device)

        # for bge embedder
        embedder = BGEEmbedder('BAAI/bge-base-en-v1.5', device, max_length)

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
            '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl'
        ]

        rank_datafile_map = {
            0: [0, 1],
            1: [2],
            2: [3],
            3: [4],
            4: [6, 7],
            5: [8],
            6: [5],
            7: [9, 10]
        }

        rank_datafiles = [datafiles[i] for i in rank_datafile_map[rank]]

        sample_cnt = 0
        for f in tqdm.tqdm(rank_datafiles):
            des_datafile = f.replace('train.jsonl', 'filtered_train.jsonl')
            paired_data = []
            for l in tqdm.tqdm(open(f, 'r').readlines()):
                paired_data.append(l)
            filtered_data = deduplicate_filter(paired_data)
            filtered_data = language_filter(filtered_data, ['en'])
            filtered_data = consistency_filter(filtered_data, embedder)

            if len(filtered_data) == 0:
                continue
        
            with open(des_datafile, 'w') as fw:
                for s, p in filtered_data:
                    fw.write(p+'\n')
                    sample_cnt += 1

        print(f'>>> total sample count: {sample_cnt}')
    else:
        filter_datafiles = [
            '/fs-computility/llm/chenzhi/datasets_processed/ELI5/filtered_train.jsonl', 
            '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/filtered_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/filtered_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI/filtered_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/Quora/filtered_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/filtered_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/filtered_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/filtered_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/filtered_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/filtered_train.jsonl',
            '/fs-computility/llm/chenzhi/datasets_processed/FEVER/filtered_train.jsonl'
        ]

        lower_bound_score = 0.7
        sample_cnt = 0
        total_cnt = 0
        analysis_mode = True
        for f in tqdm.tqdm(filter_datafiles):
            des_datafile = f.replace('filtered_train.jsonl', 'filtered_phase2_train.jsonl')
            paired_data = []
            lines = open(f, 'r').readlines()
            total_cnt += len(lines)
            for l in tqdm.tqdm(lines):
                l = json.loads(l)
                question = l['question']
                response = l['response']
                if 'cosine_score' in l:
                    bge_score = float(l['cosine_score'])
                    if question.lower() == response.lower():
                        continue

                    if bge_score < lower_bound_score:
                        break

                    paired_data.append(json.dumps(l))
                else:
                    paired_data.append(json.dumps(l))

            if len(paired_data) == 0:
                continue
        
            if analysis_mode:
                sample_cnt += len(paired_data)
            else:
                with open(des_datafile, 'w') as fw:
                    for p in paired_data:
                        fw.write(p+'\n')
                        sample_cnt += 1

        print(f'>>> filtered sample count: {sample_cnt}')
        print(f'>>> total sample count: {total_cnt}')
        print(f'>>> filtering rate: {(total_cnt - sample_cnt) / total_cnt}')
