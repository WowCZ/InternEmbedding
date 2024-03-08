import math
import json
import tqdm
import torch
import chromadb
from embedding.models.modeling_bge import BGEEmbedder
from chromadb import Documents, EmbeddingFunction, Embeddings


class BGEFunction(EmbeddingFunction):
    def __init__(self, bge_name, *args, **kwargs):
        super(BGEFunction, self).__init__(*args, **kwargs)
        self.bge_name = bge_name
        print(f'>>> Loading BGE embedder from {self.bge_name}')
        self.embedder = BGEEmbedder(self.bge_name, device='cuda', max_length=512)

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents
        doc_cnt = len(input)
        batch_size = 128 * 12
        batch_num = math.ceil(doc_cnt / batch_size)

        doc_embeddings = []
        for bi in tqdm.tqdm(range(batch_num)):
            embedding = self.embedder.encode(input[bi*batch_size : (bi+1)*batch_size])
            doc_embeddings.append(embedding)
        return torch.cat(doc_embeddings, dim=0).numpy().tolist()


def create_gaokao_chromadb(gaokao_file: str='/fs-computility/llm/shared/leizhikai/chenzhi/zh-exam-k12/detail_prompt/kindergarten_sft.jsonl'):
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_or_create_collection(name="gaokao", embedding_function=BGEFunction(bge_name='BAAI/bge-base-zh-v1.5'))

    # client.delete_collection(name="gaokao")
    # collection = client.create_collection(name="gaokao", embedding_function=BGEFunction(), metadata={"hnsw:space": "cosine"})
    # collection = client.get_collection(name="gaokao")

    gaokao = []
    gaokao_metadatas = []
    gaokao_ids = []
    with open(gaokao_file, 'r') as fr:
        for li, l in enumerate(fr.readlines()):
            l = json.loads(l)
            gaokao.append(l['prompt'])
            gaokao_metadatas.append({
                'answer': l['output'],
                'grade_class': l['grade_class'],
                'major': l['major'],
                'area': l['area'],
                'language': l['language'],
                'keypoint': ' | '.join(l['keypoint']) if all(k is not None for k in l['keypoint']) else '',
                'hard_level': l['hard_level'],
                'q_type': l['q_type']
            })
            gaokao_ids.append(f'gaokao_id{li}')

            if li > 0 and li % 41665 == 0:
                collection.add(
                    documents=gaokao,
                    metadatas=gaokao_metadatas,
                    ids=gaokao_ids
                )

                gaokao = []
                gaokao_metadatas = []
                gaokao_ids = []

                print(f'>>> Proceed {li+1} instanses!')

    if len(gaokao) > 0:
        collection.add(
            documents=gaokao,
            metadatas=gaokao_metadatas,
            ids=gaokao_ids
        )

    query_texts = collection.query(
        query_texts=["中学生小红因连续旷课，被学校处分。从法定义务的角度看，小红没有"],
        n_results=2
    )

    print(query_texts)


def create_internembedder_chromadb():
    # client = chromadb.HttpClient(host='localhost', port=8002)
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