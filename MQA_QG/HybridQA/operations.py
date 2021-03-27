"""
Define atomic operations in HybridQA. 
"""

from .utils import *
from config import qg_nlp
from nltk.tokenize import sent_tokenize, word_tokenize

'''
Convert a text question into a description
'''
def convert_question_into_desc(qa_pair):
    predicate = get_predicate(qa_pair['question'])    
    return predicate

'''
Randomly select an entity with text as bridge entity
@Return: a list of candidate bridge entities. 
''' 
def get_bridge_entities(sample):
    cell_with_links = []
    for r_ind, row in enumerate(sample['table']['data']):
        for c_ind, col in enumerate(row):
            for e_ind, link in enumerate(col[1]):
                if (link is not None) and (link in sample['text']):
                    bridge_entity = {
                        'loc': [r_ind, c_ind, e_ind], 
                        'name': col[0], 
                        'url': link
                    }
                    cell_with_links.append(bridge_entity)
    
    return cell_with_links

'''
Get the passage of the bridge entity. 
'''
def get_passage(sample, url, num_sent = 2):
    bridge_entity_text = sample['text'][url] 
    # trick: get the first two sentences
    sentences = sent_tokenize(bridge_entity_text)
    bridge_entity_text = None
    if len(sentences) >= num_sent:
        bridge_entity_text = ' '.join(sentences[0:num_sent])
    elif len(sentences) > 0:
        bridge_entity_text = sample['text'][url]
    
    return bridge_entity_text

'''
Check the vadility of each generated question (the question should contain the bridge entity). 
If question is valid, return its delexcicalized form (replacing the bridge entity with @@ENT@@)
Rule: if question contains bridge_entity_name_in_text or bridge_entity_name_in_table, then directly replace it. 
else: call extract_key_struc_from_question
'''
def filter_generated_questions(QA_pairs, table_entity_name, replace_with_ENT = True):
    valid_questions = []
    bridge_entity_name_in_text, bridge_entity_name_in_table = table_entity_name['text'], table_entity_name['table']

    for qa_pair in QA_pairs:
        if not bridge_entity_name_in_text is None:
            if qa_pair['question'].find(bridge_entity_name_in_text) >= 0:
                ques = qa_pair['question']
                if replace_with_ENT == True:
                    ques = qa_pair['question'].replace(bridge_entity_name_in_text, '@@ENT@@', 1)
                valid_questions.append({'question': ques, 'answer': qa_pair['answer']})
                continue
    
        if qa_pair['question'].find(bridge_entity_name_in_table) >= 0:
            ques = qa_pair['question']
            if replace_with_ENT == True:
                ques = qa_pair['question'].replace(bridge_entity_name_in_table, '@@ENT@@', 1)
            valid_questions.append({'question': ques, 'answer': qa_pair['answer']})
            continue

        entity_phrase, _ = extract_key_struc_from_question(qa_pair['question'])
        if not entity_phrase is None:
            if phrase_overlap(entity_phrase, bridge_entity_name_in_table) == True:
                ques = qa_pair['question']
                if replace_with_ENT == True:
                    ques = qa_pair['question'].replace(entity_phrase, '@@ENT@@', 1)
                valid_questions.append({'question': ques, 'answer': qa_pair['answer']})
                continue
            elif not bridge_entity_name_in_text is None and phrase_overlap(entity_phrase, bridge_entity_name_in_text) == True:
                ques = qa_pair['question']
                if replace_with_ENT == True:
                    ques = qa_pair['question'].replace(entity_phrase, '@@ENT@@', 1)
                valid_questions.append({'question': ques, 'answer': qa_pair['answer']})
                continue
    return valid_questions

'''
linearlize the table content
'''
def linearlize_subtable_fullrow(sample, bridge_entity):
    table = sample['table']
    # print(table)
    bridge_entity_loc = bridge_entity['loc']
    bridge_entity_name = bridge_entity['name']

    header_list = []
    for head in table['header']:
        # print(head)
        header_list.append(head[0])
        # header_list.append(', '.join([ele[0] for ele in head if ele[0] is not None]))
    table_data = []
    for row in table['data']:
        # row_list = [', '.join(col[0]) for col in row]
        row_list = [col[0] for col in row]
        table_data.append(row_list)

    prefix = 'The table title is {} . '.format(table['title'])
    tmp = ""
    for c_ind, cell in enumerate(table_data[bridge_entity_loc[0]]):
        header_text = header_list[c_ind]
        tmp += 'The {} is {} . '.format(header_text, cell)
        
    postfix = 'Start describing {} : '.format(bridge_entity_name)
    return prefix + tmp + postfix

'''
Filter table descriptions
remove those descriptions that do not start with the bridge entity name
if valid, convert it into its delexcicalized form (replacing the bridge entity with "the ____ that")
'''
def filter_table_descriptions(descriptions, bridge_entity_name_in_table):
    valid_descriptions = []
    for des in descriptions:
        if des.lower().startswith(bridge_entity_name_in_table.lower()):
            new_des = 'the ____ that' + des[len(bridge_entity_name_in_table):]
            # new_des = des.replace(bridge_entity_name_in_table, 'the ____ that', 1)
            valid_descriptions.append(new_des)
    return valid_descriptions

'''
For each entity with text, return its name in table and in text
{'text': bridge_entity_name_in_text, 'table': bridge_entity_name_in_table}
'''
def get_all_entity_names(sample, cell_with_links):
    entity_names = []
    for entity in cell_with_links:
        entity_name_in_table = entity['name']
        entity_text_url = entity['url']

        entity_text = get_passage(sample, entity_text_url)
        if entity_text is None:
            continue

        entity_name_in_text, _ = get_first_NER(entity_text)
        if entity_name_in_text is None or entity_name_in_table is None:
            continue

        if not phrase_overlap(entity_name_in_text, entity_name_in_table) == True:
            continue

        entity['text_name'] = entity_name_in_text
        entity['table_name'] = entity_name_in_table
        entity_names.append(entity)
    
    return entity_names

'''
Check if a QA pair contains contains a specific table entity
'''
def is_ques_contains_entity(qa_pair, table_entity_name):
    bridge_entity_name_in_text, bridge_entity_name_in_table = table_entity_name['text_name'], table_entity_name['table_name']

    if qa_pair['question'].find(bridge_entity_name_in_text) >= 0:
        ques = qa_pair['question'].replace(bridge_entity_name_in_text, '@@ENT2@@', 1)
        return {'question': ques, 'answer': qa_pair['answer']}
    
    if qa_pair['question'].find(bridge_entity_name_in_table) >= 0:
        ques = qa_pair['question'].replace(bridge_entity_name_in_table, '@@ENT2@@', 1)
        return {'question': ques, 'answer': qa_pair['answer']}

    entity_phrase, _ = extract_key_struc_from_question(qa_pair['question'])
    if not entity_phrase is None:
        if phrase_overlap(entity_phrase, bridge_entity_name_in_table) == True:
            ques = qa_pair['question'].replace(entity_phrase, '@@ENT@@', 1)
            return {'question': ques, 'answer': qa_pair['answer']}
        elif not bridge_entity_name_in_text is None and phrase_overlap(entity_phrase, bridge_entity_name_in_text) == True:
            ques = qa_pair['question'].replace(entity_phrase, '@@ENT@@', 1)
            return {'question': ques, 'answer': qa_pair['answer']}
    
    return None

'''
Check if a table description contains contains a specific table entity
'''
def is_desc_contains_entity(text, table_entity_name):
    bridge_entity_name_in_text, bridge_entity_name_in_table = table_entity_name['text_name'], table_entity_name['table_name']

    if text.find(bridge_entity_name_in_text) >= 0:
        text = text.replace(bridge_entity_name_in_text, '@@ENT2@@', 1)
        return text
    
    if text.find(bridge_entity_name_in_table) >= 0:
        text = text.replace(bridge_entity_name_in_table, '@@ENT2@@', 1)
        return text

    return None