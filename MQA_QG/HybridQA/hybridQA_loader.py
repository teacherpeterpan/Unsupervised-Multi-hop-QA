"""
Tools for loading hybridQA data
"""
import json
import os
from tqdm import tqdm

class HybridQA_Dataset():
    def __init__(self, config):
        print('Loading HybridQA >>>>>>>>')
        self.table_dir = config.table_dir
        self.text_dir = config.text_dir

        # get all table ids
        self.table_id_list = []
        g = os.walk(self.table_dir)
        for _, _, file_list in g:
            for file_name in file_list:
                self.table_id_list.append(file_name.replace('.json', ''))
        self.table_num = len(self.table_id_list)

        # load data
        self.dataset = {}
        for table_id in tqdm(self.table_id_list):
            self.dataset[table_id] = self.load_data_sample(table_id)
        print('Data loaded. Number of tables: {}'.format(self.table_num))

    def load_data_sample(self, table_id):
        with open(self.table_dir + table_id + '.json', 'r') as f:
            table_sample = json.load(f)
    
        with open(self.text_dir + table_id + '.json', 'r') as f:
            text_sample = json.load(f)

        return {'table_id': table_id, 'table': table_sample, 'text': text_sample}
