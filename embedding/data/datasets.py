import json
import random
import math
import bisect
from typing import List
from torch.utils.data import Dataset
from embedding.data.data_utils import extract_dataset_configs, extract_and_validate_datasets

random.seed(20)

def custom_corpus_prompt(query_prompt: str, task_type: str):
    if task_type in ['Clustering', 'Classification']:
        corpus_prompt = 'The corresponding category'
    elif task_type in ['Retrieval', 'PairClassification', 'Summarization', 'Reranking']:
        corpus_prompt = 'The corresponding document text'
    else:
        corpus_prompt = query_prompt

    return corpus_prompt

def prompt_wrapper(question: str, response: str, negatives: List[str], task_type: str, prompt_candidates: List[str]):
    q_prompt = random.choice(prompt_candidates)
    r_prompt = custom_corpus_prompt(q_prompt, task_type)

    question = q_prompt + ': ' + question
    response = r_prompt + ': ' + response
    if negatives is not None:
        if type(negatives) is not list:
            if type(negatives) is not str:
                raise TypeError(f'>>> Unknow negative sample types: {type(negatives)}')
            else:
                negatives = [negatives]
        negatives = [r_prompt + ': ' + n for n in negatives]
    
    return question, response, negatives


class EmbedderDatasets(Dataset):
    def __init__(self, datatset_config: str, task_prompt: bool=False, negative_num: int=3):
        qa_pairs = []
        dataset_infos = extract_dataset_configs(datatset_config)
        for dataset_name, dataset_info in dataset_infos.items():
            with open(dataset_info['disk_path'], 'r') as fr:
                lines = fr.readlines()
                dataset_infos[dataset_name]['sample_cnt'] = len(lines)
                sampling_cnt = math.ceil(dataset_info['sampling_ratio'] * dataset_infos[dataset_name]['sample_cnt'])

                dataset_infos[dataset_name]['sample_cnt'] = sampling_cnt
                for li, l in enumerate(lines):
                    if li > sampling_cnt:
                        break

                    qa_pairs.append((dataset_name, l))

        self.dataset_infos = dataset_infos
        
        self.qa_pairs = qa_pairs
        self.qa_num = len(qa_pairs)
        self.dataset_infos['TotalTrainingNum'] = self.qa_num
        self.task_prompt = task_prompt
        self.negative_num = negative_num

    def __len__(self):
        return self.qa_num
    
    def __str__(self) -> str:
        return json.dumps(self.dataset_infos, indent=4)

    def make_sample(self, dataset_name, sample):
        sample = json.loads(sample)

        task_type = self.dataset_infos[dataset_name]['task_type']

        if 'negative_response' not in sample:
            sample['negative_response'] = None
        else:
            sample['negative_response'] = sample['negative_response'][:self.negative_num]

        question, response, negatives = sample['question'], sample['response'], sample['negative_response']

        if self.task_prompt:
             question, response, negatives = prompt_wrapper(question, 
                                                            response, 
                                                            negatives, 
                                                            task_type, 
                                                            self.dataset_infos[dataset_name]['prompts'])

        return (question, response, negatives, task_type)

    def __getitem__(self, index):
        dataset_name, qa_sample = self.qa_pairs[index]
        return self.make_sample(dataset_name, qa_sample)
    

class EmbedderIndependentDataset(Dataset):
    def __init__(self, dataset_info: dict, task_prompt: bool=False, negative_num: int=3):
        qa_pairs = []
        with open(dataset_info['disk_path'], 'r') as fr:
            lines = fr.readlines()

            dataset_info['sample_cnt'] = len(lines)
            sampling_cnt = math.ceil(dataset_info['sampling_ratio'] * dataset_info['sample_cnt'])
            
            for li, l in enumerate(lines):
                if li > sampling_cnt:
                    break
                
                ll = json.loads(l)
                if 'negative_response' in ll and len(ll['negative_response']) < negative_num:
                    continue

                qa_pairs.append(l)

        dataset_info['sample_cnt'] = len(qa_pairs)
        self.dataset_info = dataset_info
        
        self.qa_pairs = qa_pairs
        self.qa_num = len(qa_pairs)
        self.task_prompt = task_prompt
        self.negative_num = negative_num

    def __len__(self):
        return self.qa_num
    
    def __str__(self) -> str:
        return json.dumps(self.dataset_info, indent=4)

    def make_sample(self, sample):
        sample = json.loads(sample)

        task_type = self.dataset_info['task_type']

        if 'negative_response' not in sample:
            sample['negative_response'] = None
        else:
            sample['negative_response'] = sample['negative_response'][:self.negative_num]

        question, response, negatives = sample['question'], sample['response'], sample['negative_response']
        
        if self.task_prompt:
            question, response, negatives = prompt_wrapper(question, 
                                                           response, 
                                                           negatives, 
                                                           task_type, 
                                                           self.dataset_info['prompts'])
            
        return (question, response, negatives, self.dataset_info['task_type'])

    def __getitem__(self, index):
        qa_sample = self.qa_pairs[index]
        return self.make_sample(qa_sample)
    

class EmbedderConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datatset_config: str, task_prompt: bool=False, negative_num: int=3):
        super().__init__()
        datasets = []
        dataset_infos = extract_dataset_configs(datatset_config)
        # dataset_infos, invalidated_datasets = extract_and_validate_datasets(datatset_config)

        for dataset_info in dataset_infos.values():
            datasets.append(
                EmbedderIndependentDataset(dataset_info, task_prompt, negative_num)
            )

        self.datasets = datasets

        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]

        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __str__(self) -> str:
        dataset_infos = [d.dataset_info for d in self.datasets]
        return json.dumps(dataset_infos, indent=4)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes

