"""
Tools for loading hotpotQA data
"""
import json
import os
from tqdm import tqdm

class HotpotQA_Dataset():
    def __init__(self, config):
        print('Loading HotpotQA >>>>>>>>')
        self.data_dir = config.data_path
        # load data
        with open(self.data_dir, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        self.dataset = {sample['passage_id'] : sample['context'] for sample in data_list}
        print('Data loaded. Number of examples: {}'.format(len(self.dataset)))



