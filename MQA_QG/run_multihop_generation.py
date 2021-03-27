from config import *

from HybridQA.hybridQA_loader import HybridQA_Dataset
from HybridQA.reasoning import *

from HotpotQA.hotpotQA_loader import HotpotQA_Dataset
from HotpotQA.reasoning import *

import json
from tqdm import tqdm

def construct_sample(q_id, question, table_id, q_type, multi_ans_node = False):
    data_sample = {}
    data_sample['table_id'] = str(table_id)
    data_sample['question'] = question['question']
    data_sample['answer-text'] = question['answer_text'].strip()
    # note: answer node is a list of list!
    data_sample['answer-node'] = [[question['bridge_entity'],
                                question['bridge_entity_loc'],
                                question['bridge_entity_text_url'], q_type]]
    if multi_ans_node == True:
        data_sample['answer-node'] = []
        for i in range(len(question['bridge_entity'])):
            data_sample['answer-node'].append([[question['bridge_entity'][i],
                                question['bridge_entity_loc'][i],
                                question['bridge_entity_text_url'][i], q_type]])
    
    data_sample['question_id'] = str(q_id)
    data_sample['where'] = q_type
    q_doc = stanza_nlp(question['question'])
    data_sample['question_postag'] = ' '.join([w.xpos for w in q_doc.sentences[0].words])
    return data_sample

def generate_for_HybridQA(args):
    hybridQA = HybridQA_Dataset(args)

    data_range_start, data_range_end = DATA_RANGE
    global_id = data_range_start
    all_tables = sorted(list(hybridQA.dataset.items()), key=lambda x: x[0])
    
    if data_range_end == -1:
        all_tables = all_tables[data_range_start:]
    else:
        all_tables = all_tables[data_range_start:data_range_end]

    all_questions_json = []
    for table_id, sample_table in tqdm(all_tables):
        ques_list = None
        if QUESTION_TYPE == 'table2text':
            ques_list = Generate_Table_to_Text_Question(sample_table, q_number=QUESTION_NUM)
        elif QUESTION_TYPE == 'text2table':
            ques_list = Generate_Text_to_Table_Question(sample_table, q_number=QUESTION_NUM)
        elif QUESTION_TYPE == 'text_only':
            ques_list = Generate_Text_Only_Question(sample_table, q_number=QUESTION_NUM)
        elif QUESTION_TYPE == 'table_only':
            ques_list = Generate_Table_Only_Question(sample_table, q_number=QUESTION_NUM)
        else:
            raise(NotImplementedError)
        
        if not ques_list is None:
            for ques in ques_list:
                q_type = 'table' if QUESTION_TYPE in ['text2table', 'table_only'] else 'passage'
                data_sample = construct_sample(global_id, ques, table_id, q_type)
                all_questions_json.append(data_sample)
                global_id += 1

    random.shuffle(all_questions_json)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(json.dumps(all_questions_json, indent=2))

def generate_for_HotpotQA(args):
    hotpotQA = HotpotQA_Dataset(args)
    
    all_samples = sorted(list(hotpotQA.dataset.items()), key=lambda x: x[0])
    data_range_start, data_range_end = DATA_RANGE
    if data_range_end == -1:
        all_samples = all_samples[data_range_start:]
    else:
        all_samples = all_samples[data_range_start:data_range_end]
    
    all_questions_json = []

    for passage_id, sample_passage in tqdm(all_samples):
        ques_list = None
        if QUESTION_TYPE == 'text2text':
            ques_list = Generate_Text_to_Text_Question(sample_passage, q_number = QUESTION_NUM)
        elif QUESTION_TYPE == 'comparison':
            ques_list = Generate_Comparison_Questions(sample_passage)
        else:
            raise(NotImplementedError)
        
        if not ques_list is None:
            info = {'passage_id': passage_id, "ques_ans": []}
            for ques in ques_list:
                info['ques_ans'].append({"question": ques['question'], 'answer': ques['answer_text']})
            all_questions_json.append(info)
    
    random.shuffle(all_questions_json)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(json.dumps(all_questions_json, indent=2))

if __name__ == "__main__":
    if EXPERIMENT == 'HybridQA':
        generate_for_HybridQA(parser.parse_args())
    elif EXPERIMENT == 'HotpotQA':
        generate_for_HotpotQA(parser.parse_args())