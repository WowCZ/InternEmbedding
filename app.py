import json
from apps.vector_db.text_loading_chroma import create_gaokao_chromadb, create_internembedder_chromadb
from apps.retrieval.gaokao_rag import retrieval_from_gaokao
from apps.retrieval.internembedder_rag import test_internembedder
from apps.clustering.gaokao import create_subject_keypoint_db, evaluate_subject_keypoint_match, subject_zh_en_map, extract_keypoint_embedding_data

# subject = 'mathematics'
# chromadb_path = f'/fs-computility/llm/shared/chenzhi/chromadbs/{subject}_gaokao_questions'
# gaokao_file ='/fs-computility/llm/shared/leizhikai/chenzhi/zh-exam-k12/detail_prompt/kindergarten_sft.jsonl'
# ckpt = '/fs-computility/llm/chenzhi/ckpts/bge_keypoint_triple5_20240314072748/bge_keypoint_triple5_2000.pt'
# chromabd_name = 'questions_train'
# create_gaokao_chromadb(gaokao_file, chromadb_path, chromabd_name, subject, ckpt)

# topk = 10
# saved_retrieval_file = f'/fs-computility/llm/shared/chenzhi/gaokao/{subject}_retrieval_from_keypoint_bge.jsonl'
# retrieval_from_gaokao(gaokao_file, chromadb_path, chromabd_name, subject, topk, ckpt, saved_retrieval_file)

subject = 'biology'
saved_retrieval_file = f'/fs-computility/llm/shared/chenzhi/gaokao/{subject}_retrieval_from_keypoint_bge.jsonl'

import openai
def get_llm_response(question: str):
    response = openai.ChatCompletion.create(
        api_base='http://172.28.32.57:20240/v1',
        api_key='EMPTY',
        model='internlm2-chat-20b',
        messages=[
            {"role": "user", "content": f'{question}'},
            ],
        max_tokens=4096,
        temperature=1.2,
        top_p=0.95,
        n=1,
    )
    return response['choices'][0]['message']['content']

biology_prompt = '以下是一个生物考试题的例子：\n{retrieval_q}{retrieval_a}\n 请参考以上题，回答下面具备相同知识点的题目：{q}'

with open(saved_retrieval_file, 'r') as fr:
    lines = fr.readlines()
    llen = len(lines)
    kp_match_cnt = 0
    kp_retrieval_qa = []
    for li, l in enumerate(lines):
        if li >= 50:
            break
        l = json.loads(l)
        retrieval_kps = [r['keypoint'] for r in l['retrieval']][:4]
        cur_kp = l['keypoint']

        question = l['prompt']
        answer = l['answer']

        retrieval_q = l['retrieval'][0]['prompt']
        retrieval_a = l['retrieval'][0]['answer']

        response = get_llm_response(question)
        kp_question = biology_prompt.format(retrieval_q=retrieval_q, retrieval_a=retrieval_a, q=question)
        kp_response = get_llm_response(kp_question)
        kp_retrieval_qa.append({
            'question': question,
            'llm_response': response,
            'llm_kp_response': kp_response,
            'golden': answer
        })

        if cur_kp in retrieval_kps:
            kp_match_cnt += 1
    
    print(f'>>> Recall Accuracy on Subject {subject}: {kp_match_cnt/llen}')

with open('keypoint_retrieval_response.json', 'w') as fw:
    json.dump(kp_retrieval_qa, fw, indent=4, ensure_ascii=False)



# create_internembedder_chromadb()
# test_math()
# test_internembedder()
# create_subject_keypoint_db()

# # Extract dataset
# save_dir = '/fs-computility/llm/chenzhi/datasets_processed/GAOKAO/'
# startk = 30
# hard_num = 7
# subjects = ['mathematics', 'biology',  'physics', 'chemistry', 'history']
# for subject in subjects:
#     print(f'>>> Extract {subject} embedding triples....')
#     extract_keypoint_embedding_data(subject, startk, hard_num, save_dir)
# exit(0)




# ckpt = '/fs-computility/llm/chenzhi/ckpts/bge_keypoint_triple5_20240314072748/bge_keypoint_triple5_2000.pt'
# create_subject_keypoint_db('mathematics', ckpt)

# topk = 10
# subject_statistics = dict()
# for major, subject in subject_zh_en_map.items():
#     if subject not in ['mathematics']:
#         continue

#     recall_statitics = evaluate_subject_keypoint_match(subject, topk, ckpt)
#     subject_statistics[subject] = recall_statitics
#     subject_statistics[subject]['major'] = major

# print(json.dumps(subject_statistics, indent=4, ensure_ascii=False))
# with open(f'/fs-computility/llm/chenzhi/InternEmbedding/results/gaokao/subject_top{topk}_statistics.json', 'w') as fw:
#     json.dump(subject_statistics, fw, indent=4, ensure_ascii=False)













# datafiles = [
#         '/fs-computility/llm/chenzhi/datasets_processed/STELI5/train_bge_retrieval.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/Quora/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STWikiAnswers/train_bge_retrieval.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/STAGNews/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAltlex/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAmazonReview/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STCodeSearchNet/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STFlickr30k/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STNPR/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STPAQ/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STS2ORCTA/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STXSum/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STCCNews/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTWoW/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTTrex/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTMedMCQA/train_bge_retrieval.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTPubMed/train_bge_retrieval.jsonl'
# ]

# import tqdm, json
# rank=3
# rank_datafile_map = {
#         0: list(range(0,8)),
#         1: list(range(8,16)),
#         2: list(range(16,24)),
#         3: list(range(24,33))
# }

# def score_idx(x: float) -> int:
#     if x > 0.5:
#         return 5 + score_idx(x-0.5)
    
#     if x > -1e-3 and x <= 1e-3:
#         return 0

#     if x > 1e-3 and x <= 0.1:
#         return 1

#     if x > 0.1 and x <= 0.2:
#         return 2

#     if x > 0.2 and x <= 0.3:
#         return 3

#     if x > 0.3 and x <= 0.4:
#         return 4
    
#     if x > 0.4 and x <= 0.5:
#         return 5

# rank_datafiles = [datafiles[i] for i in rank_datafile_map[rank]]

# li = 0
# hard_negative_cnt = 0
# score_statis = dict()
# for file in tqdm.tqdm(rank_datafiles):
#     save_file = file.replace('train_bge_retrieval.jsonl', 'train_bge_retrieval_triples.jsonl')
#     hard_negative_triples = []
#     fr = open(file, 'r').readlines()
#     flen = len(fr)
#     fname = file.split('/')[-2]
#     print(f'>>> {fname}: {flen}')
#     li += flen
#     tmp_hard_negative_cnt = 0
#     for l in fr:
#         l = json.loads(l)

#         gid = 0 if l['response'] == l['bge_retrieval_documents'][0] else 1
#         dedup_idx = score_idx(l['bge_distances'][gid]) # min(l['bge_distances'][:2])
#         if dedup_idx == 0:
#             continue

#         if (l['bge_distances'][2] - l['bge_distances'][gid] >= 0.05):
#             tmp_hard_negative_cnt += 1
#             hard_negative_triples.append(
#                 {
#                     'question': l['query'],
#                     'response': l['response'],
#                     'negative_response': l['bge_retrieval_documents'][2:]
#                 }
#             )
    
#         if dedup_idx not in score_statis:
#             score_statis[dedup_idx] = 0

#         score_statis[dedup_idx] += 1

#     if tmp_hard_negative_cnt == 0:
#         print(f'>>> Drop datatset {fname}')
#     hard_negative_cnt += tmp_hard_negative_cnt
#     print(f'>>> add hard negative samples: {tmp_hard_negative_cnt}')

#     with open(save_file, 'w') as fw:
#         for qr in hard_negative_triples:
#             fw.write(json.dumps(qr, ensure_ascii=False)+'\n')

# print(f'>>> total samples: {li}')
# print(f'>>> hard negative samples: {hard_negative_cnt}')
# print(json.dumps(score_statis, indent=4))

# training_datatset_files = [
#         '/fs-computility/llm/chenzhi/datasets_processed/STELI5/train_bge_retrieval_triples.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/Quora/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STWikiAnswers/train_bge_retrieval_triples.jsonl', 
#         '/fs-computility/llm/chenzhi/datasets_processed/STAGNews/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAltlex/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STAmazonReview/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STCodeSearchNet/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STFlickr30k/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STNPR/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STPAQ/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STS2ORCTA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STXSum/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/STCCNews/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTWoW/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTTrex/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTMedMCQA/train_bge_retrieval_triples.jsonl',
#         '/fs-computility/llm/chenzhi/datasets_processed/MTPubMed/train_bge_retrieval_triples.jsonl']

# import tqdm
# tlen = 0
# for file in tqdm.tqdm(training_datatset_files):
#     fr = open(file, 'r').readlines()
#     flen = len(fr)
#     fname = file.split('/')[-2]
#     if flen > 100000:
#         print(f'{fname}: {100000/flen}')
#         flen = 100000
#     tlen += flen

# print(f'>>> Total triple negative samples: {tlen}')