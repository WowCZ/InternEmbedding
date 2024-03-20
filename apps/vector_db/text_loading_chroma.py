import os
import json
import tqdm
import math
import yaml
import chromadb
from apps.embedder.bge import BGEFunction
from apps.clustering.gaokao import subject_zh_en_map

def create_gaokao_chromadb(gaokao_file: str, chromadb_path: str, chromabd_name: str, subject: str, ckpt: str):
    client = chromadb.PersistentClient(path=str(chromadb_path))
    try:
        client.delete_collection(chromabd_name)
    except:
        print(client.list_collections())

    collection = client.get_or_create_collection(name=chromabd_name, embedding_function=BGEFunction(bge_name='BAAI/bge-base-zh-v1.5', bge_ckpt=ckpt))

    gaokao = []
    gaokao_metadatas = []
    gaokao_ids = []
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
            gaokao_ids.append(f'gaokao_{cur_s}_id{li}')


    gaokao = gaokao[1500:]
    gaokao_metadatas = gaokao_metadatas[1500:]
    gaokao_ids = gaokao_ids[1500:]

    chromadb_process_limit = 41665
    for left in range(0, len(gaokao), chromadb_process_limit):
        right = left + chromadb_process_limit

        collection.add(
            documents=gaokao[left: right],
            metadatas=gaokao_metadatas[left: right],
            ids=gaokao_ids[left: right]
        )


def create_internembedder_chromadb():
    rank = int(os.environ['CUDA_VISIBLE_DEVICES'])
    with open('/fs-computility/llm/chenzhi/InternEmbedding/configs/datasets.yaml','r') as f:
        dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    dataset_root_dir = dataset_info['root_path']
    datafiles = []
    for dataset in dataset_info['internembedder_datasets']:
        dname = dataset['name']
        if dataset['task_type'] in ['Classification', 'Clustering']:
            continue
        loading_file = os.path.join(dataset_root_dir, dname, 'train.jsonl')
        if os.path.exists(loading_file):
            datafiles.append(os.path.join(dataset_root_dir, dname, 'train.jsonl'))

    rank_capacity = len(datafiles) // 8
    for ri, i in enumerate(range(0, len(datafiles), rank_capacity)):            
        if ri == rank:
            if ri < 7:
                rank_datafiles = datafiles[i: i+rank_capacity]
            else:
                rank_datafiles = datafiles[i:]
            break

    llen = 0
    for file in rank_datafiles:
        llen += len(open(file, 'r').readlines())

    print(json.dumps(rank_datafiles, indent=4))
    print('>>> Number of files: ', len(rank_datafiles))
    print('>>> Number of examples: ', llen)

    chroma_path = f'/fs-computility/llm/shared/chenzhi/chromadbs/internembedder_independent_dataset_docs_{rank}'
    client = chromadb.PersistentClient(path=str(chroma_path))
    # client.delete_collection(name="internembedder")
    collection = client.get_or_create_collection(name="internembedder", embedding_function=BGEFunction(bge_name='BAAI/bge-base-en-v1.5', bge_ckpt=None), metadata={"hnsw:space": "cosine"})

    li = 0
    docs = []
    doc_ids = []
    for file in tqdm.tqdm(rank_datafiles):
        fname = file.split('/')[-2]
        fi = 0
        for l in open(file, 'r').readlines():
            # if li <= (cur_collection_cnt-1):
            #     li += 1
            #     fi += 1
            #     continue
            
            l = json.loads(l)
            doc = l['response']
            docs.append(doc)
            doc_ids.append(f'doc_{fname}_id{fi}')

            if li > 0 and li % 41665 == 0:
                collection.add(
                    documents=docs,
                    ids=doc_ids
                )

                docs = []
                doc_ids = []

                print(f'>>> Proceed {li+1} instanses!')

            li += 1
            fi += 1


    if len(docs) > 0:
        collection.add(
            documents=docs,
            ids=doc_ids
        )
        print(f'>>> Proceed {li+1} instanses!')