import os
import math
import json
import tqdm
import torch
import chromadb
from embedding.models.modeling_bge import BGEEmbedder
from chromadb import Documents, EmbeddingFunction, Embeddings
from apps.vector_db.text_loading_chroma import BGEFunction

subject_zh_en_map = {
    '政治': 'politics',
    '思想品德(道德与法治)': 'morality_law',
    '历史': 'history',
    '语文': 'Chinese',
    '地理': 'geography',
    '数学': 'mathematics',
    '生物': 'biology',
    '物理': 'physics',
    '化学': 'chemistry',
    '英语': 'English',
    '科学': 'science',
    '历史与社会': 'history_society',
    '编程': 'program',
    '通用技术': 'general_technology',
    '信息技术': 'information_technology'
}

def keypoint_match(source: str, target: str):
    source = source.split(' | ')
    target = target.split(' | ')

    for s in source:
        if s in target:
            return True
        
    return False

def create_subject_keypoint_db(subject: str='', ckpt: str=None):
    gaokao_file: str='/fs-computility/llm/shared/leizhikai/chenzhi/zh-exam-k12/detail_prompt/kindergarten_sft.jsonl'

    gaokao_metadatas = []
    subjects = {}
    with open(gaokao_file, 'r') as fr:
        for l in fr.readlines():
            l = json.loads(l)
            keypoint = ' | '.join(l['keypoint']) if all(k is not None for k in l['keypoint']) else ''
            major = l['major']
            cur_s = subject_zh_en_map[major]
            if len(keypoint) == 0 or cur_s != subject:
                continue

            gaokao_metadatas.append({
                'question': l['prompt'],
                'answer': l['output'],
                'grade_class': l['grade_class'],
                'major': major,
                'subject': cur_s,
                'area': l['area'],
                'language': l['language'],
                'keypoint': keypoint,
                'hard_level': l['hard_level'],
                'q_type': l['q_type']
            })
            
            if cur_s not in subjects:
                subjects[cur_s] = {}

            keypoints = subjects[cur_s]
            if keypoint and keypoint not in keypoints:
                kid = len(keypoints)
                keypoints[keypoint] = f'{cur_s}_keypoint_{kid}'

    print(f'>>> Count of Subjects: {len(subjects)}')
    print(list(subjects.keys()))

    for s, keypoints in subjects.items():
        print(f'>>> Count of {s} Keypoints: {len(keypoints)}')

        chroma_path = f'/fs-computility/llm/shared/chenzhi/chromadbs/{s}_keypoints'
        client = chromadb.PersistentClient(path=str(chroma_path))
        # client.delete_collection(name="keypoints_train")
        print(client.list_collections())
        collection = client.get_or_create_collection(name="keypoints_train", embedding_function=BGEFunction(bge_name='BAAI/bge-base-zh-v1.5', bge_ckpt=ckpt), metadata={"hnsw:space": "cosine"})

        chromadb_process_limit = 41665
        keypoint_ids = list(keypoints.values())
        keypoints = list(keypoints.keys())
        for left in range(0, len(keypoints), chromadb_process_limit):
            right = left + chromadb_process_limit

            collection.add(
                documents=keypoints[left: right],
                ids=keypoint_ids[left: right]
            )


def evaluate_subject_keypoint_match(subject: str='information_technology', topk: int=1, ckpt: str=None):
    gaokao_file: str='/fs-computility/llm/shared/leizhikai/chenzhi/zh-exam-k12/detail_prompt/kindergarten_sft.jsonl'

    gaokao_metadatas = []
    with open(gaokao_file, 'r') as fr:
        for l in fr.readlines():
            l = json.loads(l)
            keypoint = ' | '.join(l['keypoint']) if all(k is not None for k in l['keypoint']) else ''
            major = l['major']
            cur_s = subject_zh_en_map[major]
            if len(keypoint) == 0 or cur_s != subject:
                continue

            gaokao_metadatas.append({
                'question': l['prompt'],
                'answer': l['output'],
                'grade_class': l['grade_class'],
                'major': major,
                'subject': cur_s,
                'area': l['area'],
                'language': l['language'],
                'keypoint': keypoint,
                'hard_level': l['hard_level'],
                'q_type': l['q_type']
            })

    chroma_path = f'/fs-computility/llm/shared/chenzhi/chromadbs/{subject}_keypoints'
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name="keypoints_train", embedding_function=BGEFunction(bge_name='BAAI/bge-base-zh-v1.5', bge_ckpt=ckpt))
    cluster_cnt = collection.count()

    match_cnt = 0
    gaokao_metadatas = gaokao_metadatas[:100]

    query_texts = collection.query(
        query_texts=[q['question'] for q in gaokao_metadatas],
        n_results=min(topk, cluster_cnt)
    )

    for qi, q in enumerate(gaokao_metadatas):
        target_keypoint = q['keypoint']
        match_status = []
        for ti in range(min(topk, cluster_cnt)):
            source_keypoint = query_texts['documents'][qi][ti]
            match_status.append(keypoint_match(source_keypoint, target_keypoint))

        if any(match_status):
            match_cnt += 1
    
    print(f'>>> Subject {subject} Keypoint Recall Accuracy Rate: {match_cnt/len(gaokao_metadatas)}')

    return {
        f'top{topk}_accuracy': match_cnt/len(gaokao_metadatas),
        'cluster_cnt': cluster_cnt
    }


def extract_keypoint_embedding_data(subject: str='information_technology', startk: int=0, hard_num: int=1, save_dir: str='.'):
    gaokao_file: str='/fs-computility/llm/shared/leizhikai/chenzhi/zh-exam-k12/detail_prompt/kindergarten_sft.jsonl'

    gaokao_metadatas = []
    with open(gaokao_file, 'r') as fr:
        for l in fr.readlines():
            l = json.loads(l)
            keypoint = ' | '.join(l['keypoint']) if all(k is not None for k in l['keypoint']) else ''
            major = l['major']
            cur_s = subject_zh_en_map[major]
            if len(keypoint) == 0 or cur_s != subject:
                continue

            gaokao_metadatas.append({
                'question': l['prompt'],
                'answer': l['output'],
                'grade_class': l['grade_class'],
                'major': major,
                'subject': cur_s,
                'area': l['area'],
                'language': l['language'],
                'keypoint': keypoint,
                'hard_level': l['hard_level'],
                'q_type': l['q_type']
            })

    
    def extract_triples_from_metadatas(collection, metadatas: list, startk: int, hard_num: int, save_dir: str, data_mode: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        metadata_cnt = len(metadatas)
        cluster_cnt = collection.count()

        embedding_triples = []
        chromadb_process_limit = 41665
        for left in range(0, metadata_cnt, chromadb_process_limit):
            right = left + chromadb_process_limit

            query_texts = collection.query(
                query_texts=[q['question'] for q in metadatas[left: right]],
                n_results=min(startk+hard_num, cluster_cnt)
            )

            for q, bge_retrievals in zip(metadatas[left: right], query_texts['documents']):
                embedding_triples.append(
                    {
                        'question': q['question'],
                        'response': q['keypoint'],
                        'negative_response': bge_retrievals[startk: startk+hard_num]
                    }
                )

        with open(os.path.join(save_dir, f'{data_mode}.jsonl'), 'w') as fw:
            for t in embedding_triples:
                fw.write(json.dumps(t)+'\n')
    
    # TODO: we can also choose the training set including all subject keypoints
    gaokao_train_metadatas = gaokao_metadatas[1500:]
    gaokao_eval_metadatas = gaokao_metadatas[1000: 1500]
    chroma_path = f'/fs-computility/llm/shared/chenzhi/chromadbs/{subject}_keypoints'
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name="keypoints", embedding_function=BGEFunction(bge_name='BAAI/bge-base-zh-v1.5', bge_ckpt=None))
    extract_triples_from_metadatas(collection, gaokao_eval_metadatas, startk, hard_num, save_dir, f'eval_{subject}')
    extract_triples_from_metadatas(collection, gaokao_train_metadatas, startk, hard_num, save_dir, f'train_{subject}')