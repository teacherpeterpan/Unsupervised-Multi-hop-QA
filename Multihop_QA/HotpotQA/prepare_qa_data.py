'''
Preprocess dataset for training SpanBERT
'''

import json
import sys
from tqdm import tqdm
import argparse

def load_passages(src_path):
    dataset_dict = {}
    with open(src_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    for p in dataset:
        dataset_dict[p['passage_id']] = (' '.join(p['context']) + ' ( yes or no )')
    return dataset_dict

def load_qas(qa_path):
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_set = json.load(f)
    return qa_set

def find_answer_char_span(context, answer_text):

    detected_answers = [{'answer_text': answer_text, 'char_spans': []}]
    if answer_text.lower() == 'yes':
        yes_no_index = context.find('( yes or no )')
        detected_answers[0]['char_spans'] = [[yes_no_index + 2, yes_no_index + 2 + 2]]
    elif answer_text.lower() == 'no':
        yes_no_index = context.find('( yes or no )')
        detected_answers[0]['char_spans'] = [[yes_no_index + 9, yes_no_index + 9 + 1]]
    else:
        flag = False
        if context.count(answer_text) == 0:
            answer_text_ = answer_text.rstrip(".,/;:'[]()`+-=_<>?·@$#%^&*!").strip()
            if len(answer_text_) == len(answer_text) or len(answer_text_) == 0 or answer_text.count('.'):
                answer_text_ = answer_text
                a = answer_text.split()
                aa = []
                flag = True
                for i, w in enumerate(a):
                    if w.count('-') and len(w) > 1:
                        w = w.split('-')
                        w = ' - '.join(w)
                        aa += w.split()
                        # aa += [w[0], '-', w[1]] if w[1] != 'year' else [w[0] + '-' + w[1]]
                    elif w.count('.') and len(w) > 1:
                        w = w.split('.')
                        w = ' . '.join(w)
                        aa += w.split()
                    elif w.count('/') and len(w) > 1:
                        w = w.split('/')
                        w = ' / '.join(w)
                        aa += w.split()
                    elif w.rstrip('!') == '':
                        aa += [ww for ww in w]
                    else:
                        aa += [w]
                answer_text = ' '.join(aa)
            else:
                answer_text = answer_text_
        
        if flag:
            a = aa

        if context.count(answer_text) == 0 or len(answer_text) < 1:
            return None

        detected_answers = [{'answer_text': answer_text, 'char_spans': []}]
        while context.count(answer_text) > 0:
            index = context.index(answer_text)
            detected_answers[0]['char_spans'] += [[index, index + len(answer_text) - 1]]
            context = context[index + len(answer_text):]

    if len(detected_answers[0]['char_spans']) == 0:
        return None
    
    return detected_answers

def data_processing(src_path, qa_path, output_path, q_type = 'all'):
    passage_dict = load_passages(src_path)
    qa_set = load_qas(qa_path)

    examples, cnt = [], 0

    for sample in tqdm(qa_set):
        p_id = sample['passage_id']
        if p_id in passage_dict and len(sample['ques_ans']) > 0:
            example = {'context': passage_dict[p_id], 'qas': []}
            for qa in sample['ques_ans']:
                info = {'qid': cnt, 'question': qa['question'], 'detected_answers': []}
                detected_answers = find_answer_char_span(example['context'], qa['answer'])
                if detected_answers is None:
                    continue
                info['detected_answers'] = detected_answers
                if q_type == 'all':
                    example['qas'].append(info)
                    cnt += 1
                elif q_type == 'bridge':
                    if qa['type'] == 'bridge':
                        example['qas'].append(info)
                        cnt += 1
                elif q_type == 'comparison':
                    if qa['type'] == 'comparison':
                        example['qas'].append(info)
                        cnt += 1
                else:
                    continue
            
            if len(example['qas']) > 0:
                examples.append(json.dumps(example))
    
    print('Number of valid examples: {}'.format(len(examples)))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(examples))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", default=None, type=str, required=True,
                        help="Path of HotpotQA source data.")
    parser.add_argument("--qa_path", default=None, type=str, required=True,
                        help="Path of HotpotQA QA data.")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="Output data path.")
    args = parser.parse_args()

    data_processing(args.src_path, args.qa_path, args.output_path)
    
            


        




