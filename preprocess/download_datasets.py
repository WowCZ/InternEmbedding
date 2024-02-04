import os
import json
import tqdm
import random
from datasets import load_dataset

random.seed(20)

HFCACHEDATASETS = '/fs-computility/llm/chenzhi/datasets_cache'

def download_eli5_datasets(dataset_name: str, save_dir: str):
    def get_question(example):
        title = example["title"]
        selftext = example["selftext"]
        if selftext:
            if selftext[-1] not in [".", "?", "!"]:
                seperator = ". "
            else:
                seperator = " "
            question = title + seperator + selftext
        else:
            question = title
        example["question"] = question
        return example

    dataset = load_dataset(dataset_name, cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    dataset = dataset.map(get_question)

    qa_pairs = []
    for d in dataset['train']:
        question, preference_answer = d['question'], d['response_j']
        qa_pairs.append(json.dumps(
            {
                'question': question,
                'response': preference_answer
            }
        )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qa_pairs))


def download_hotpotqa_datasets(dataset_name: str, save_dir: str):
    def get_related_sents(example):
        related_sentence_ids = example["supporting_facts"]["sent_id"]
        related_titles = example["supporting_facts"]["title"]
        related_title_sentence_map = dict()
        for title, sid in zip(related_titles, related_sentence_ids):
            if title not in related_title_sentence_map:
                related_title_sentence_map[title] = []
            related_title_sentence_map[title].append(sid)

        context_titles = example["context"]["title"]
        context_sentences = example["context"]["sentences"]
        related_sents = []
        for title, sentences in zip(context_titles, context_sentences):
            if title in related_title_sentence_map:
                related_sents.append(''.join(sentences))

        example["related_sents"] = related_sents
        return example

    dataset = load_dataset(dataset_name, 'fullwiki', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    dataset = dataset.map(get_related_sents)

    qp_pairs = []
    for d in dataset['train']:
        question, related_sents = d['question'], d['related_sents']
        for sent in related_sents:
            qp_pairs.append(json.dumps(
                {
                    'question': question,
                    'response': sent
                }
            )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_msmarco_datasets(dataset_name: str, save_dir: str):
    def get_related_passages(example):
        related_passage_ids = [i for i, select in enumerate(example["passages"]["is_selected"]) if select == 1]
        related_passages = [example["passages"]["passage_text"][si] for si in related_passage_ids]

        negative_passage_ids = [i for i, select in enumerate(example["passages"]["is_selected"]) if select == 0]
        negative_passages = [example["passages"]["passage_text"][si] for si in negative_passage_ids]

        example["related_passages"] = related_passages
        example["negative_passages"] = negative_passages
        return example

    subsets = ['v1.1', 'v2.1']
    qp_pairs = []
    for subset in subsets:
        dataset = load_dataset(dataset_name, subset, cache_dir=HFCACHEDATASETS, trust_remote_code=True)
        dataset = dataset.map(get_related_passages)
        for d in dataset['train']:
            question, related_passages, negative_passages = d['query'], d['related_passages'], d['negative_passages']
            # for passage in related_passages:
            if len(related_passages) > 0 and len(negative_passages) > 0:
                qp_pairs.append(json.dumps(
                    {
                        'question': question,
                        'response': random.choice(related_passages),
                        'negative_response': random.choice(negative_passages)
                    }
                )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_multinli_datasets(dataset_name: str, save_dir: str):
    qp_pairs = []
    dataset = load_dataset(dataset_name, cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    for d in dataset['train']:
        if d['label'] != 0:
            continue

        premise, hypothesis = d['premise'], d['hypothesis']
        qp_pairs.append(json.dumps(
            {
                'question': premise,
                'response': hypothesis
            }
        )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_quora_datasets(dataset_name: str, save_dir: str):
    qp_pairs = []
    dataset = load_dataset(dataset_name, cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    for d in dataset['train']:
        if not d['is_duplicate']:
            continue

        q1, q2 = d['questions']['text']
        qp_pairs.append(json.dumps(
            {
                'question': q1,
                'response': q2
            }
        )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_miracl_datasets(dataset_name: str, save_dir: str):
    subsets = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']
    qp_pairs = []
    for subset in subsets:
        dataset = load_dataset(dataset_name, subset, split='train', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
        for di, d in enumerate(dataset):
            if di > 100000:
                break
            query, positive_passages = d['query'], d['positive_passages']
            for passage in positive_passages:
                qp_pairs.append(json.dumps(
                    {
                        'question': query,
                        'response': passage['text']
                    }
                )+'\n')
        print(f'>>> Successfully processing on subset {subset}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_mrtydi_datasets(dataset_name: str, save_dir: str):
    subsets = ['arabic', 'bengali', 'english', 'finnish', 'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai']
    qp_pairs = []
    for subset in subsets:
        dataset = load_dataset(dataset_name, subset, cache_dir=HFCACHEDATASETS, trust_remote_code=True)
        for d in dataset['train']:
            query, positive_passages = d['query'], d['positive_passages']
            for passage in positive_passages:
                qp_pairs.append(json.dumps(
                    {
                        'question': query,
                        'response': passage['text']
                    }
                )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_nq_datasets(dataset_name: str, save_dir: str):
    subsets = ['default']
    qp_pairs = []
    for subset in subsets:
        dataset = load_dataset(dataset_name, subset, cache_dir=HFCACHEDATASETS, trust_remote_code=True)
        for d in tqdm.tqdm(dataset['train']):
            query, document = d['question']['text'], d['document']['tokens']
            document_content = ' '.join([t for t, html in zip(document['token'], document['is_html']) if not html])
            qp_pairs.append(json.dumps(
                {
                    'question': query,
                    'response': document_content
                }
            )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_squad_datasets(dataset_name: str, save_dir: str):
    qp_pairs = []
    dataset = load_dataset(dataset_name, cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    for d in dataset['train']:
        question, context = d['question'], d['context']
        qp_pairs.append(json.dumps(
            {
                'question': question,
                'response': context
            }
        )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_triviaqa_datasets(dataset_name: str, save_dir: str):
    qp_pairs = []
    dataset = load_dataset(dataset_name, 'rc', split='train', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    for d in dataset:
        question, search_results = d['question'], d['search_results']
        search_ranks = search_results['rank']
        search_max_rank_ids = [ri for ri, r in enumerate(search_ranks) if r == max(search_ranks)]
        selected_searches = [search_results['search_context'][ri] for ri in search_max_rank_ids]
        for search in selected_searches:
            qp_pairs.append(json.dumps(
                {
                    'question': question,
                    'response': search
                }
            )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_fever_datasets(dataset_name: str, save_dir: str):
    cw_pairs = []
    dataset = load_dataset(dataset_name, 'v1.0', split='train', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    for d in dataset:
        claim, evidence_wiki_url, label = d['claim'], d['evidence_wiki_url'], d['label']
        if label in ['SUPPORTS', 'REFUTES']:
            cw_pairs.append((claim, evidence_wiki_url))

    wikipages = load_dataset(dataset_name, 'wiki_pages', split='wikipedia_pages', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    wikipage_map = dict()
    for page in wikipages:
        wikipage_map[page['id']] = page['text']

    qp_pairs = []
    for claim, evidence_wiki_url in tqdm.tqdm(cw_pairs):
        try:
            qp_pairs.append(json.dumps(
                {
                    'question': claim,
                    'response': wikipage_map[evidence_wiki_url]
                }
            )+'\n')
        except:
            print(f'>>> cannot find wiki url like {evidence_wiki_url}')
            continue

    qp_pairs = list(set(qp_pairs))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_dureader_datasets(dataset_name: str, save_dir: str):
    qp_pairs = []
    dataset = load_dataset(dataset_name, split='train', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    for d in dataset:
        question, context = d['question'], d['context']
        qp_pairs.append(json.dumps(
            {
                'question': question,
                'response': context
            }
        )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_t2ranking_datasets(dataset_name: str, save_dir: str):
    qp_pairs = []
    # qp_id_pairs = load_dataset(dataset_name, 'qrels.retrieval.train', split='train', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    collection = load_dataset(dataset_name, 'collection', split='train', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    queries = load_dataset(dataset_name, 'queries.train', split='train', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    
    qp_id_pairs = dict()
    hard_sample = load_dataset(dataset_name, 'train.mined.tsv', split='train', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    for qid, pid, score in zip(hard_sample['qid'], hard_sample['pid'], hard_sample['score']):
        if qid not in qp_id_pairs:
            qp_id_pairs[qid] = []
        qp_id_pairs[qid].append((pid, score))

    for qid, ps in qp_id_pairs.items():
        qp_id_pairs[qid] = sorted(ps, key=lambda x: x[1], reverse=True)[0][0]

    query_map = dict()
    for qid, text in zip(queries['qid'], queries['text']):
        query_map[qid] = text

    passage_map = dict()
    for pid, text in zip(collection['pid'], collection['text']):
        passage_map[pid] = text

    for qid, pid in qp_id_pairs.items():
        qp_pairs.append(json.dumps(
            {
                'question': query_map[qid],
                'response': passage_map[pid]
            }
        )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_negation_datasets(dataset_name: str, save_dir: str):
    qp_pairs = []
    dataset = load_dataset(dataset_name, split='train', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    for d in dataset:
        question, context, hard_negative = d['anchor'], d['entailment'], d['negative']
        qp_pairs.append(json.dumps(
            {
                'question': question,
                'response': context,
                'negative_response': hard_negative
            }
        )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_gooqa_datasets(dataset_name: str, save_dir: str):
    # git clone https://github.com/allenai/gooaq.git
    # git lfs pull
    qp_pairs = []
    with open(dataset_name, 'r') as fr:
        for l in fr.readlines():
            l = json.loads(l)

            response = None
            if l['short_answer']:
                response = l['short_answer']

            if l['answer']:
                response = l['answer']

            if response:
                qp_pairs.append(json.dumps(
                    {
                        'question': l['question'],
                        'response': response,
                    }
                )+'\n')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))

    print('>>> GooQA: ', len(qp_pairs))


def download_yahooqa_datasets(dataset_name: str, save_dir: str):
    qp_pairs = []
    dataset = load_dataset(dataset_name, split='train', cache_dir=HFCACHEDATASETS, trust_remote_code=True)
    for d in dataset:
        question, context = d['question'], d['answer']
        qp_pairs.append(json.dumps(
            {
                'question': question,
                'response': context,
            }
        )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


def download_stembedding_datasets(dataset_name: str, save_dir: str):
    qp_pairs = []
    
    with open(dataset_name, 'r') as fr:
        for l in fr.readlines():
            l = json.loads(l)
            
            qp_pairs.append(json.dumps(
                {
                    'question': l[0],
                    'response': l[1]
                }
            )+'\n')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as fw:
        fw.write(''.join(qp_pairs))


if __name__ == '__main__':
    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/ELI5'
    # download_eli5_datasets('vincentmin/eli5_rlhf', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA'
    # download_hotpotqa_datasets('hotpot_qa', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO'
    # download_msmarco_datasets('ms_marco', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI'
    # download_multinli_datasets('multi_nli', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/Quora'
    # download_quora_datasets('quora', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/MIRACL'
    # download_miracl_datasets('miracl/miracl', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi'
    # download_mrtydi_datasets('castorini/mr-tydi', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions'
    # download_nq_datasets('natural_questions', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/SQuAD'
    # download_squad_datasets('squad', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA'
    # download_triviaqa_datasets('trivia_qa', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/FEVER'
    # download_fever_datasets('fever', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/DuReader'
    # download_dureader_datasets('PaddlePaddle/dureader_robust', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking'
    # download_t2ranking_datasets('THUIR/T2Ranking', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO_Triple'
    # download_msmarco_datasets('ms_marco', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/JinaAINegation'
    # download_negation_datasets('jinaai/negation-dataset', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/GooQA'
    # download_gooqa_datasets('/fs-computility/llm/chenzhi/datasets_cache/gooaq/data/gooaq.jsonl', save_dir)

    # save_dir = '/fs-computility/llm/chenzhi/datasets_processed/YahooQA'
    # download_yahooqa_datasets('yahoo_answers_qa', save_dir)

    dataset_dirs = ['st_allnli', 'st_eli5', 'st_gooqa', 'st_specter', 'st_stackexchange_dup', 'st_wikihow', 'st_yahoo_qa']
    save_dirs = ['STAllNLI', 'STELI5', 'STGooQA', 'STSpecter', 'STStackexchangeDup', 'STWikiHow', 'STYahooQA']

    for dataset, save_dir in tqdm.tqdm(zip(dataset_dirs, save_dirs)):
        dataset_root = f'/fs-computility/llm/chenzhi/datasets_cache/{dataset}'
        for _, _, fs in os.walk(dataset_root):
            for f in fs:
                if f.endswith('.jsonl'):
                    dataset_name = os.path.join(dataset_root, f)
                    if os.path.exists(dataset_name):
                        print('>>> download from ', dataset_name)
                        download_stembedding_datasets(dataset_name, f'/fs-computility/llm/chenzhi/datasets_processed/{save_dir}')
