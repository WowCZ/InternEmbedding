import os
import json
import math
import yaml
import tqdm
import random
from collections import OrderedDict
from typing import Sized, Iterator, List
from torch.utils.data import Sampler, BatchSampler

random.seed(20)

def extract_dataset_configs(config_file: str):
    with open(config_file,'r') as f:
        dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    dataset_root_dir = dataset_info['root_path']
    extracted_dataset_info = OrderedDict()
    for dataset in dataset_info['internembedder_datasets']:
        dname = dataset['name']

        disk_path = os.path.join(dataset_root_dir, dname, 'train.jsonl')
        if not os.path.exists(disk_path):
            print(f'>>> Loadining dataset {dname} failed, where the disk path {disk_path} does not exist.')
            continue

        while dname in extracted_dataset_info:
            if '-' not in dname:
                dname = dname + '-1'
            else:
                num = int(dname.split('-')[-1]) + 1
                dname = dname + str(num)

        extracted_dataset_info[dname] = {
            'task_type': dataset['task_type'],
            'prompts': dataset['prompts'],
            'sampling_ratio': dataset['sampling_ratio'],
            'disk_path': disk_path
        }
    
    return extracted_dataset_info

def extract_and_validate_datasets(dataset_config: str):
    dataset_infos = extract_dataset_configs(dataset_config)

    invalidated_datasets = []
    for dataset_name, dataset_info in tqdm.tqdm(dataset_infos.items()):
        with open(dataset_info['disk_path'], 'r') as fr:
            lines = fr.readlines()

            for l in lines:
                l = json.loads(l)
                if dataset_info['task_type'] in ['Clustering', 'Classification']:
                    if 'negative_response' not in l or len(l['negative_response']) == 0:
                        print(json.dumps(l, indent=4, ensure_ascii=False))
                        print(f'>>> {dataset_name} is Invalidated!')
                        invalidated_datasets.append(dataset_name)
                        break
                else:
                    if 'negative_response' in l and len(l['negative_response']) == 0:
                        print(json.dumps(l, indent=4, ensure_ascii=False))
                        print(f'>>> {dataset_name} is Invalidated!')
                        invalidated_datasets.append(dataset_name)
                        break
    return dataset_infos, invalidated_datasets


class InDatasetSampler(Sampler):
    def __init__(self, data_source: Sized, batch_size: int = 1) -> None:
        super().__init__(data_source)

        cumulative_sizes = data_source.cumulative_sizes

        batched_dataset_indxs = dict()
        for di in range(len(cumulative_sizes)):
            if di == 0:
                left = 0
            else:
                left = cumulative_sizes[di-1]
            
            right = cumulative_sizes[di]

            dataset_idxs = list(range(left, right))

            random.shuffle(dataset_idxs)

            for bl in range(0, len(dataset_idxs), batch_size):
                batched_dataset_indxs[dataset_idxs[bl]] = dataset_idxs[bl: bl+batch_size]

        self.batched_dataset_indxs = batched_dataset_indxs

        self.search_indxs = list(self.batched_dataset_indxs.keys())
        random.shuffle(self.search_indxs)

    def __iter__(self) -> Iterator[int]:
        return iter(self.search_indxs)

    def __len__(self) -> int:
        return len(self.search_indxs)
    

class InDatasetBatchSampler(BatchSampler):
    def __init__(self, sampler: Sampler[int], batch_size, *args, **kwargs) -> None:
        super().__init__(sampler, batch_size=batch_size, drop_last=False)

        assert type(sampler) is InDatasetSampler

        self.batched_dataset_indxs = sampler.batched_dataset_indxs

    def __iter__(self) -> Iterator[List[int]]:
        sampler_iter = iter(self.sampler)

        for bi in sampler_iter:
            batch = self.batched_dataset_indxs[bi]
            yield batch

    def __len__(self) -> int:
        return len(self.sampler)