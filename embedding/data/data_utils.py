import os
import yaml
from collections import OrderedDict

def extract_dataset_configs(config_file: str):
    with open(config_file,'r') as f:
        dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    dataset_root_dir = dataset_info['root_path']
    extracted_dataset_info = OrderedDict()
    for dataset in dataset_info['internembedder_datasets']:
        dname = dataset['name']
        origin_dname = dataset['name']

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
            'disk_path': os.path.join(dataset_root_dir, origin_dname, 'train.jsonl')
        }
    
    return extracted_dataset_info