import json
from apps.vector_db.text_loading_chroma import create_gaokao_chromadb, create_internembedder_chromadb
from apps.retrieval.gaokao_rag import retrieval_from_gaokao, retrieval_from_raw_questions
from apps.retrieval.internembedder_rag import retrieval_for_internembedder_datasets
from apps.clustering.gaokao import create_subject_keypoint_db, evaluate_subject_keypoint_match, subject_zh_en_map, extract_keypoint_embedding_data

subject = 'biology'
chromadb_path = f'/fs-computility/llm/shared/chenzhi/chromadbs/{subject}_gaokao_questions'
# gaokao_file = '/fs-computility/llm/shared/chenzhi/gaokao/xueersi_v0327_dedup.jsonl'
gaokao_file ='/fs-computility/llm/shared/leizhikai/chenzhi/zh-exam-k12/detail_prompt/kindergarten_sft.jsonl'
# ckpt = '/fs-computility/llm/chenzhi/ckpts/bge_keypoint_triple5_20240314072748/bge_keypoint_triple5_2000.pt'
ckpt = '/fs-computility/llm/chenzhi/ckpts/bge_gaokao_xes_kp_20240411155015/bge_gaokao_xes_kp_1281.pt'
# ckpt = None
chromabd_name = 'questions_train_xes_kg'

create_gaokao_chromadb(gaokao_file, chromadb_path, chromabd_name, subject, ckpt)

topk = 10
retrieval_gaokao_file ='/fs-computility/llm/shared/yangyf/share/random_sample_trained_wrong_case_on_train_set.jsonl'
saved_retrieval_file = f'/fs-computility/llm/shared/chenzhi/gaokao/train_hard_{subject}_retrieval_from_scratch_bge.jsonl'
retrieval_from_gaokao(retrieval_gaokao_file, chromadb_path, chromabd_name, subject, topk, ckpt, saved_retrieval_file)

# # saved_retrieval_file = f'/fs-computility/llm/shared/chenzhi/gaokao/{subject}_raw_retrieval_from_keypoint_bge.jsonl'
# # retrieval_from_raw_questions(questions, chromadb_path, chromabd_name, topk, ckpt, saved_retrieval_file)

# saved_retrieval_file = f'/fs-computility/llm/shared/chenzhi/gaokao/{subject}_retrieval_from_keypoint_bge.jsonl'
# llm_name = 'internlm2-chat-20b'

# import openai
# def get_llm_response(question: str):
#     response = openai.ChatCompletion.create(
#         api_base='http://172.28.0.81:20240/v1',
#         api_key='EMPTY',
#         model=llm_name,
#         messages=[
#             {"role": "user", "content": f'{question}'},
#             ],
#         max_tokens=2048,
#         temperature=1.2,
#         top_p=0.95,
#         n=1,
#     )
#     return response['choices'][0]['message']['content']

# biology_prompt = '题目一：\n{retrieval_q}\n回答：{retrieval_a}\n 题目二：\n{q}\n回答：'
# evaluation_prompt = '以下是一个化学考试题和对应的标准答案： \n问题：{question}\n标准答案：{answer}\n有来自于两个不同模型的关于该问题的回答：\n回复一：{response} \n 回复二：{kp_response}\n参考标准答案上面两个回复有以下四种情况：\n（1）两个都对; （2）只有回复一对；（3）只有回复二对；（4）两个都错。\n请不带任何解释地直接输出以上情况的编号：（'

# with open(saved_retrieval_file, 'r') as fr:
#     lines = fr.readlines()
#     llen = len(lines)
#     kp_match_cnt = 0
#     kp_retrieval_qa = []
#     for li, l in enumerate(lines):

#         if li >= 100:
#             break
#         print(f'>>> Processing {subject} sample: {li}')

#         l = json.loads(l)
#         retrieval_kps = [r['keypoint'] for r in l['retrieval']][:4]
#         cur_kp = l['keypoint']

#         question = l['prompt']
#         answer = l['answer']

#         retrieval_q = l['retrieval'][0]['prompt']
#         retrieval_a = l['retrieval'][0]['answer']

#         question = question.replace('\n', '')
#         answer = answer.replace('\n', '')

#         retrieval_q = retrieval_q.replace('\n', '')
#         retrieval_a = retrieval_a.replace('\n', '')

#         response = get_llm_response(question)
#         retrieval_r = get_llm_response(retrieval_q)
#         kp_question = biology_prompt.format(retrieval_q=retrieval_q, retrieval_a=retrieval_a, q=question)
#         kp_response = get_llm_response(kp_question)

#         response = response.replace('\n', '')
#         kp_response = kp_response.replace('\n', '')
#         eval_input = evaluation_prompt.format(question=question, answer=answer, response=response, kp_response=kp_response)
#         eval_ans = get_llm_response(eval_input)

#         kp_retrieval_qa.append({
#             'question': question,
#             'llm_response': response,
#             'llm_kp_response': kp_response,
#             'golden': answer,
#             'llm_preference': eval_ans,
#             'retrieval_q': retrieval_q,
#             'retrieval_a': retrieval_a,
#             'retrieval_r': retrieval_r
#         })

#         if cur_kp in retrieval_kps:
#             kp_match_cnt += 1
    
#     print(f'>>> Recall Accuracy on Subject {subject}: {kp_match_cnt/llen}')

# with open(f'/fs-computility/llm/chenzhi/InternEmbedding/results/gaokao/{subject}_keypoint_retrieval_{llm_name}_format_response10.json', 'w') as fw:
#     json.dump(kp_retrieval_qa, fw, indent=4, ensure_ascii=False)