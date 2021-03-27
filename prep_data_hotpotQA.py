'''
Build HotpotQA dataset from its raw data
'''
import argparse
import json
from tqdm import tqdm

def load_hotpotQA(data_path):
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def title_match(evidences, title):
    for eve in evidences:
        if title.strip().find(eve.strip()) >=0:
            return True
    return False 

'''
1. select only evidence passages
output: train/dev.src.txt ; train/dev.qa.txt
'''
def build_dataset(data_path, output_path, data_split):
    print(f'Building {data_split} dataset...')
    dataset = load_hotpotQA(data_path)
    eve_passages = []
    QAs = []
    for sample in tqdm(dataset):
        # add QA
        qa_pairs = {'passage_id': sample['_id'], 
                    'ques_ans': [
                        {'question': sample['question'], 'answer': sample['answer'], 'type': sample['type']}
                    ],
                    }
        # add passage
        evidences = set([ele[0] for ele in sample['supporting_facts']])
        evidence_passages = []
        for passage in sample['context']:
            title, sentences = passage[0], passage[1]
            if title_match(evidences, title) == True:
                article = ' '.join([sent.strip() for sent in sentences])
                evidence_passages.append(article)
        
        if len(evidence_passages) >= 2:
            eve_passages.append({'passage_id': sample['_id'], 'context': evidence_passages})
            QAs.append(qa_pairs)
    print('Number of samples in {}: {}'.format(data_split, len(eve_passages)))
    
    with open(f'{output_path}{data_split}.src.json', 'w') as f:
        f.write(json.dumps(eve_passages, indent=2))

    with open(f'{output_path}{data_split}.qa.json', 'w') as f:
        f.write(json.dumps(QAs, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=None, type=str, required=True,
                        help="Path of HotpotQA train set.")
    parser.add_argument("--dev_dir", default=None, type=str, required=True,
                        help="Path of HotpotQA dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Output data path.")
    args = parser.parse_args()

    build_dataset(args.train_dir, args.output_dir, 'train')
    build_dataset(args.dev_dir, args.output_dir, 'dev')
