import json
import chromadb
from apps.embedder.bge import BGEFunction
from apps.clustering.gaokao import subject_zh_en_map


def retrieval_from_gaokao(gaokao_file: str, chromadb_path: str, chromabd_name: str, subject: str, topk: int, ckpt: str, saved_retrieval_file: str):
    client = chromadb.PersistentClient(path=str(chromadb_path))
    collection = client.get_collection(name=chromabd_name, embedding_function=BGEFunction(bge_name='BAAI/bge-base-zh-v1.5', bge_ckpt=ckpt))
    question_cnt = collection.count()
    
    gaokao = []
    gaokao_metadatas = []
    with open(gaokao_file, 'r') as fr:
        for li, l in enumerate(fr.readlines()):
            l = json.loads(l)

            major = l['major']
            keypoint = ' | '.join(l['keypoint']) if all(k is not None for k in l['keypoint']) else ''
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
                'keypoint': ' | '.join(l['keypoint']) if all(k is not None for k in l['keypoint']) else '',
                'hard_level': l['hard_level'],
                'q_type': l['q_type']
            })

    gaokao = gaokao[:1000]
    gaokao_metadatas = gaokao_metadatas[:1000]
    
    query_texts = collection.query(
            query_texts=gaokao,
            n_results=min(topk, question_cnt))
    
    retrieved_gaokao_metadatas = []
    for qi, q in enumerate(gaokao_metadatas):
        retrieval_metadates = query_texts['metadatas'][qi]
        retrieval_questions = query_texts['documents'][qi]
        
        for pi, p in enumerate(retrieval_questions):
            retrieval_metadates[pi]['prompt'] = p
            
        q['retrieval'] = retrieval_metadates
        retrieved_gaokao_metadatas.append(q)
    
    with open(saved_retrieval_file, 'w') as fw:
        for q in retrieved_gaokao_metadatas:
            fw.write(json.dumps(q)+'\n')