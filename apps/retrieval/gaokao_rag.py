import json
import chromadb
from typing import List
from apps.embedder.bge import BGEFunction
from apps.clustering.gaokao import subject_zh_en_map


def retrieval_from_gaokao(gaokao_file: str, chromadb_path: str, chromabd_name: str, subject: str, topk: int, ckpt: str, saved_retrieval_file: str):
    client = chromadb.PersistentClient(path=str(chromadb_path))
    collection = client.get_collection(name=chromabd_name, embedding_function=BGEFunction(bge_name='BAAI/bge-base-zh-v1.5', bge_ckpt=ckpt))
    
    gaokao = []
    gaokao_metadatas = []
    with open(gaokao_file, 'r') as fr:
        for li, l in enumerate(fr.readlines()):
            l = json.loads(l)

            major = l['major']
            if type(l['keypoint']) is not str:
                keypoint = ' | '.join(l['keypoint']) if all(k is not None for k in l['keypoint']) else ''
            else:
                keypoint = l['keypoint']

            cur_s = subject_zh_en_map[major]
            if len(keypoint) == 0 or cur_s != subject:
                continue

            gaokao.append(l['prompt'])
            gaokao_metadatas.append({
                'prompt': l['prompt'],
                'answer': l['output'],
                'grade_class': l['grade_class'],
                'major': l['major'],
                'subject': cur_s,
                'area': l['area'],
                'language': l['language'],
                'keypoint': keypoint,
                'hard_level': l['hard_level'],
                'q_type': l['q_type']
            })

    retrieved_gaokao_metadatas = retrieval_from_question(collection, topk, gaokao, gaokao_metadatas)
    
    with open(saved_retrieval_file, 'w') as fw:
        for q in retrieved_gaokao_metadatas:
            fw.write(json.dumps(q)+'\n')


def retrieval_from_question(collection, topk: int, questions: List[str], question_metadatas: List[dict]):
    question_cnt = collection.count()

    chromadb_process_limit = 41665
    query_texts = {
        'metadatas': [],
        'documents': []
    }
    for left in range(0, len(questions), chromadb_process_limit):
        right = left + chromadb_process_limit

        cur_query_texts = collection.query(
                query_texts=questions[left: right],
                n_results=min(topk, question_cnt))
        query_texts['metadatas'].extend(cur_query_texts['metadatas'])
        query_texts['documents'].extend(cur_query_texts['documents'])

    retrieved_gaokao_metadatas = []
    for qi in range(len(questions)):
        retrieval_metadates = query_texts['metadatas'][qi]
        retrieval_questions = query_texts['documents'][qi]
        
        for pi, p in enumerate(retrieval_questions):
            retrieval_metadates[pi]['prompt'] = p
        
        if question_metadatas is not None:
            question_metadatas[qi]['retrieval'] = retrieval_metadates
            retrieved_gaokao_metadatas.append(question_metadatas[qi])
        else:
            retrieved_gaokao_metadatas.append({
                'prompt': questions[qi],
                'retrieval': retrieval_metadates
            })

    return retrieved_gaokao_metadatas


def retrieval_from_raw_questions(questions: List[str], chromadb_path: str, chromabd_name: str, topk: int, ckpt: str, saved_retrieval_file: str):
    client = chromadb.PersistentClient(path=str(chromadb_path))
    collection = client.get_collection(name=chromabd_name, embedding_function=BGEFunction(bge_name='BAAI/bge-base-zh-v1.5', bge_ckpt=ckpt))
    
    retrieved_metadatas = retrieval_from_question(collection, topk, questions, None)
    
    with open(saved_retrieval_file, 'w') as fw:
        for q in retrieved_metadatas:
            fw.write(json.dumps(q)+'\n')