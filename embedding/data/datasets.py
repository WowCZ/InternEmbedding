import json
import random
import math
from typing import List
from torch.utils.data import Dataset
from embedding.data.data_utils import dataset_sampling_ratios, dataset_task_prompts

random.seed(20)

class EmbedderDatasets(Dataset):
    def __init__(self, datatset_files: List[str], task_prompt: bool=False):
        qa_pairs = []
        dataset_statistics = dict()
        for datatset_file in datatset_files:
            dataset_name = datatset_file.split('/')[-2]
            with open(datatset_file, 'r') as fr:
                lines = fr.readlines()
                dataset_statistics[dataset_name] = len(lines)
                if dataset_name in dataset_sampling_ratios:
                    sampling_cnt = math.ceil(dataset_sampling_ratios[dataset_name] * dataset_statistics[dataset_name])
                else:
                    sampling_cnt = dataset_statistics[dataset_name]

                dataset_statistics[dataset_name] = sampling_cnt
                for li, l in enumerate(lines):
                    if li > sampling_cnt:
                        break

                    qa_pairs.append((dataset_name, l))

        self.dataset_statistics = dataset_statistics
        
        self.qa_pairs = qa_pairs
        self.qa_num = len(qa_pairs)
        self.dataset_statistics['TotalTrainingQAPairs'] = self.qa_num
        self.task_prompt = task_prompt

    def __len__(self):
        return self.qa_num
    
    def __str__(self) -> str:
        return json.dumps(self.dataset_statistics, indent=4)

    def make_sample(self, dataset_name, sample):
        sample = json.loads(sample)
        if 'negative_response' not in sample:
            sample['negative_response'] = None
        question, response, negatives = sample['question'], sample['response'], sample['negative_response']
        if self.task_prompt:
            sampled_prompt = random.choice(dataset_task_prompts[dataset_name])
            question = sampled_prompt + ': ' + question
            response = sampled_prompt + ': ' + response
            if negatives is not None:
                if type(negatives) is not list:
                    if type(negatives) is not str:
                        raise TypeError(f'>>> Unknow negative sample types: {type(negatives)}')
                    negatives = [negatives]
                negatives = [sampled_prompt + ': ' + n for n in negatives]
            
        return (question, response, negatives)

    def __getitem__(self, index):
        dataset_name, qa_sample = self.qa_pairs[index]
        return self.make_sample(dataset_name, qa_sample)
