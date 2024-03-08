import math
import json
import tqdm
import torch
import chromadb
from embedding.models.modeling_bge import BGEEmbedder
from chromadb import Documents, EmbeddingFunction, Embeddings
from apps.vector_db.text_loading_chroma import BGEFunction


def test_internembedder():
    import os
    rank = int(os.environ['CUDA_VISIBLE_DEVICES'])
    chroma_path = f'/fs-computility/llm/shared/chenzhi/chromadb_{rank}'
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name="internembedder", embedding_function=BGEFunction(bge_name='BAAI/bge-base-en-v1.5'))

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
    queries = []
    responses = []
    doc_ids = []
    for file in tqdm.tqdm(rank_datafiles):
        fname = file.split('/')[-2]
        fi = 0
        bge_credit_corpus = []
        save_file = file.replace('train.jsonl', 'train_bge_retrieval.jsonl')
        if os.path.exists(save_file):
            continue

        for l in open(file, 'r').readlines():
            l = json.loads(l)
            query = l['question']
            response = l['response']
            doc_id = f'doc_{fname}_id{fi}'

            queries.append(query)
            responses.append(response)
            doc_ids.append(doc_id)

            if li > 0 and li % 41665 == 0:
                query_retrieval_texts = collection.query(
                        query_texts=queries,
                        n_results=7
                )

                for i, (gid, rids) in enumerate(zip(doc_ids, query_retrieval_texts['ids'])):
                    if gid in rids[:2]:
                        qr = {
                            'topk_ids': query_retrieval_texts['ids'][i],
                            'bge_distances': query_retrieval_texts['distances'][i],
                            'bge_retrieval_documents': query_retrieval_texts['documents'][i],
                            'query': queries[i],
                            'response': responses[i]
                        }
                        bge_credit_corpus.append(qr)

                queries = []
                responses = []
                doc_ids = []

            li += 1
            fi += 1
        
        if len(queries) > 0:
            query_retrieval_texts = collection.query(
                    query_texts=queries,
                    n_results=7
            )

            for i, (gid, rids) in enumerate(zip(doc_ids, query_retrieval_texts['ids'])):
                if gid in rids[:2]:
                    qr = {
                        'topk_ids': query_retrieval_texts['ids'][i],
                        'bge_distances': query_retrieval_texts['distances'][i],
                        'bge_retrieval_documents': query_retrieval_texts['documents'][i],
                        'query': queries[i],
                        'response': responses[i]
                    }
                    bge_credit_corpus.append(qr)

            queries = []
            responses = []
            doc_ids = []


        print(f'>>> BGE filtering rate: {len(bge_credit_corpus)}/{fi}={len(bge_credit_corpus)/fi}')
        with open(save_file, 'w') as fw:
            for qr in bge_credit_corpus:
                fw.write(json.dumps(qr, ensure_ascii=False)+'\n')