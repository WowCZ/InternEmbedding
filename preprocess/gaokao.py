import os
import tqdm
import glob
import json
import random


def gaokao_exam():
    with open("/fs-computility/llm/chenzhi/InternEmbedding/resets/gaokao/single_choices_prompt.txt","r") as f:
        single_choice_labels = [l.strip() for l in f.readlines()]

    with open("/fs-computility/llm/chenzhi/InternEmbedding/resets/gaokao/multi_choices_prompt_nomajor.txt","r") as f:
        mutiple_choice_labels = [l.strip() for l in f.readlines()]

    problems = []

    for root, directories, files in os.walk('/fs-computility/llm/shared/leizhikai/chenzhi/zh-exam-k12/standard'):
            for filename in files:
                filepath = os.path.join(root, filename)
                with open(filepath,"r") as f:
                    for p in f.readlines():
                        p = json.loads(p)
                        if len(p['std_ans']) == 1:
                            p['q_type'] = "单选题"
                            labels = single_choice_labels
                        else:
                            assert len(p['std_ans']) > 1
                            p['q_type'] = "多选题"
                            labels = mutiple_choice_labels

                        prompt_idx = random.randint(0,len(labels)-1)
                        prompt = labels[prompt_idx]

                        # Check if the current prompt ends with punctuation
                        if prompt.endswith(('。', '？', '！', '：', '；', '，', '、')):
                            problem = prompt + p['q_main'] + "\n"
                        else:
                            if random.random() < 0.5:
                                problem = prompt + ' ' + p['q_main'] + "\n"
                            else:
                                problem = prompt + '\n' + p['q_main'] + "\n"

                        for i,option in enumerate(p['options']):
                            problem += chr(ord('A')+i) + ". " + option + "\n"
                            
                        solve = ". ".join(p['std_ans'])+". " + "\n" + p['answer_detail']
                        problems.append(json.dumps({
                            'prompt': problem,
                            'output': solve.strip(),
                            'grade_class': p['grade_class'],
                            'major': p['major'],
                            'area': p['area'],
                            'language': p['language'],
                            'keypoint': p['keypoint'],
                            'hard_level': p['hard_level'],
                            'q_type': p['q_type']
                        },ensure_ascii=False)+"\n")

    with open("/fs-computility/llm/shared/leizhikai/chenzhi/zh-exam-k12/detail_prompt/kindergarten_sft.jsonl","w") as f:
        f.writelines(problems)


def extract_xes_keypoint(keypoint: dict):
    if keypoint is None:
        return []

    def dfs(keypoint):
        if len(keypoint) == 0:
            return ['']
        
        all_items = []
        for k, v in keypoint.items():
            k = k.rstrip('🆗')
            all_items.extend([k+'-'+cur_kp for cur_kp in dfs(v)])

        return all_items
    
    all_items = dfs(keypoint)
    
    all_items = [(kp, len(kp.split('-'))) for kp in all_items if kp.find('知识点') != -1]
    all_items = sorted(all_items, key=lambda x:x[1], reverse=True)

    if len(all_items) > 0:
        max_kl = all_items[0][1]
        all_items = list(set([kp.rstrip('-') for (kp, kl) in all_items if kl == max_kl]))

    return all_items


def gaokao_xes():
    with open("/fs-computility/llm/chenzhi/InternEmbedding/resets/gaokao/single_choices_prompt.txt","r") as f:
        single_choice_labels = [l.strip() for l in f.readlines()]

    with open("/fs-computility/llm/chenzhi/InternEmbedding/resets/gaokao/multi_choices_prompt_nomajor.txt","r") as f:
        mutiple_choice_labels = [l.strip() for l in f.readlines()]

    xes_path = '/fs-computility/llm/shared/leizhikai/kaoshi/xueersi/v0327_dedup/'
    problems = []
    for file in tqdm.tqdm(glob.glob(rf'{xes_path}*')):
        with open(file, 'r') as fr:
            for p in fr.readlines():
                p = json.loads(p)
                if len(p['std_ans']) == 1:
                    p['q_type'] = "单选题"
                    labels = single_choice_labels
                else:
                    continue
                    # assert len(p['std_ans']) > 1
                    # p['q_type'] = "多选题"
                    # labels = mutiple_choice_labels

                keypoint = extract_xes_keypoint(p['keypoint'][0])

                prompt_idx = random.randint(0,len(labels)-1)
                prompt = labels[prompt_idx]

                # Check if the current prompt ends with punctuation
                if prompt.endswith(('。', '？', '！', '：', '；', '，', '、')):
                    problem = prompt + p['q_main'] + "\n"
                else:
                    if random.random() < 0.5:
                        problem = prompt + ' ' + p['q_main'] + "\n"
                    else:
                        problem = prompt + '\n' + p['q_main'] + "\n"

                for i,option in enumerate(p['options']):
                    problem += chr(ord('A')+i) + ". " + option + "\n"
                    
                solve =  p['answer_detail'] + '\n' + '答案：' + ". ".join(p['std_ans'])+". "
                problems.append(json.dumps({
                    'prompt': problem,
                    'output': solve.strip(),
                    'grade_class': p['grade_class'],
                    'major': p['major'],
                    'area': p['area'],
                    'language': p['language'],
                    'keypoint': ' | '.join(keypoint),
                    'hard_level': p['hard_level'],
                    'q_type': p['q_type']
                },ensure_ascii=False)+"\n")

    with open("/fs-computility/llm/shared/chenzhi/gaokao/xueersi_v0327_dedup.jsonl","w") as f:
        f.writelines(problems)


def change_to_embedding_training(file: str):

    fw = open("/fs-computility/llm/shared/chenzhi/internembedding_datasets/XESGAOKAO/train.jsonl", "a")
    with open(file, 'r') as fr:
        for p in tqdm.tqdm(fr.readlines()):
            p = json.loads(p)
            if p['keypoint'] is not None and len(p['keypoint']) > 0:
                keypoints = p['keypoint'].split(' | ')
            else:
                continue

            for kp in keypoints:
                fw.write(json.dumps({
                    'question': p['prompt'],
                    'response': kp
                },ensure_ascii=False)+"\n")
                fw.flush()

    fw.close()

change_to_embedding_training("/fs-computility/llm/shared/chenzhi/gaokao/xueersi_v0327_dedup.jsonl")