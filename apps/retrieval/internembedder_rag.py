import os
import json
import tqdm
import yaml
import chromadb
from apps.embedder.bge import BGEFunction

def retrieval_for_internembedder_datasets(dataset_config: str, chroma_path: str, collection: str):
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
    collection = client.get_collection(name='internembedder', embedding_function=BGEFunction(bge_name='BAAI/bge-base-en-v1.5', bge_ckpt=None))

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
                        n_results=100
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