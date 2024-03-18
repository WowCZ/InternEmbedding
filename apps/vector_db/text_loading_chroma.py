import json
import tqdm
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
    import os
    rank = int(os.environ['CUDA_VISIBLE_DEVICES'])
    chroma_path = f'/fs-computility/llm/shared/chenzhi/chromadb_{rank}'
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name="internembedder", embedding_function=BGEFunction(bge_name='BAAI/bge-base-en-v1.5'), metadata={"hnsw:space": "cosine"})

    datafiles = [
        '/fs-computility/llm/chenzhi/datasets_processed/STELI5/train.jsonl', 
        '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/Quora/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STWikiAnswers/train.jsonl', 
        '/fs-computility/llm/chenzhi/datasets_processed/STAGNews/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STAltlex/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STAmazonReview/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STCodeSearchNet/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STFlickr30k/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STNPR/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STPAQ/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STS2ORCTA/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STXSum/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/STCCNews/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MTWoW/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MTTrex/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MTMedMCQA/train.jsonl',
        '/fs-computility/llm/chenzhi/datasets_processed/MTPubMed/train.jsonl'
    ]
    
    rank_datafile_map = {
            0: list(range(0,8)),
            1: list(range(8,16)),
            2: list(range(16,24)),
            3: list(range(24,33))
    }

    rank_datafiles = [datafiles[i] for i in rank_datafile_map[rank]]

    li = 0
    docs = []
    doc_ids = []
    for file in tqdm.tqdm(rank_datafiles):
        fname = file.split('/')[-2]
        fi = 0
        for l in open(file, 'r').readlines():
            if li <= 6166420:
                li += 1
                fi += 1
                continue
            
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