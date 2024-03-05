import math
import json
import tqdm
import torch
import chromadb
from embedding.models.modeling_bge import BGEEmbedder
from chromadb import Documents, EmbeddingFunction, Embeddings


class BGEFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        doc_cnt = len(input)
        batch_size = 128 * 8
        batch_num = math.ceil(doc_cnt / batch_size)

        embedder = BGEEmbedder('BAAI/bge-base-zh-v1.5', device='cuda', max_length=512)

        doc_embeddings = []
        for bi in tqdm.tqdm(range(batch_num)):
            embedding = embedder.encode(input[bi*batch_size : (bi+1)*batch_size])
            doc_embeddings.append(embedding)
        return torch.cat(doc_embeddings, dim=0).numpy().tolist()


def create_gaokao_chromadb(gaokao_file: str='/fs-computility/llm/shared/leizhikai/chenzhi/zh-exam-k12/detail_prompt/kindergarten_sft.jsonl'):
    client = chromadb.HttpClient(host='localhost', port=8000)

    client.delete_collection(name="gaokao")
    collection = client.create_collection(name="gaokao", embedding_function=BGEFunction(), metadata={"hnsw:space": "cosine"})
    # collection = client.get_collection(name="gaokao")

    gaokao = []
    gaokao_metadatas = []
    gaokao_ids = []
    with open(gaokao_file, 'r') as fr:
        for li, l in enumerate(fr.readlines()):
            if li <= 166660:
                continue
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