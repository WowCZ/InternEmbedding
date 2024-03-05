import os
import json
import random

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
