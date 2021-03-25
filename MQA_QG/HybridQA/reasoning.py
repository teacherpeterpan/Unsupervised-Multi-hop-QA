"""
Perform random reasoning for each reasoning chain. 
"""
from .operations import *
import random
from config import qg_nlp, stanza_nlp, table_to_text, bert_fill_blank
import itertools
from .utils import *
from nltk.tokenize import word_tokenize

def ask_questions_in_text(sample, bridge_entity, num_sent = 2, replace_with_ENT = True):
    bridge_entity_name_in_table = bridge_entity['name']
    bridge_entity_text_url = bridge_entity['url']

    bridge_entity_text = get_passage(sample, bridge_entity_text_url, num_sent)
    if bridge_entity_text is None:
        return []

    bridge_entity_name_in_text, _ = get_first_NER(bridge_entity_text)
    if bridge_entity_name_in_text is None or bridge_entity_name_in_table is None:
        return []

    if not phrase_overlap(bridge_entity_name_in_text, bridge_entity_name_in_table) == True:
        bridge_entity_name_in_text = None

    table_entity_name = {'text': bridge_entity_name_in_text, 'table': bridge_entity_name_in_table}
    QA_pairs = qg_nlp.qg_without_answer(bridge_entity_text)
    filtered_QA_pairs = filter_generated_questions(QA_pairs, table_entity_name, replace_with_ENT)
    return filtered_QA_pairs

# Ask question in text for the bridge entity (the bridge entity is the answer)
def ask_question_for_bridge_entity(sample, bridge_entity):
    bridge_entity_name_in_table = bridge_entity['name']
    bridge_entity_text_url = bridge_entity['url']

    bridge_entity_text = get_passage(sample, bridge_entity_text_url)
    if bridge_entity_text is None:
        return []

    # get bridge entity name in text
    answer_text = None
    if bridge_entity_text.find(bridge_entity_name_in_table) >= 0:
        answer_text = bridge_entity_name_in_table
    else:
        entity_name, _ = get_first_NER(bridge_entity_text)
        if entity_name is None or (phrase_overlap(entity_name, bridge_entity_name_in_table) == False):
            return []
        answer_text = entity_name

    QA_pairs = qg_nlp.qg_with_answer_text(bridge_entity_text, answer_text)
    return QA_pairs
    
'''
Generate a table-to-text question
'''
def Generate_Table_to_Text_Question(sample, q_number = 3):
    # Step 1: randomly select a bridge entity
    cell_with_links = get_bridge_entities(sample)
    
    if len(cell_with_links) == 0:
        return None
    
    bridge_entity = random.sample(cell_with_links, 1)[0]
    bridge_entity_name_in_table = bridge_entity['name']
    bridge_entity_text_url = bridge_entity['url']
    bridge_entity_loc = bridge_entity['loc']

    # Step 2: Asking questions in the text
    QA_pairs = ask_questions_in_text(sample, bridge_entity)
    if len(QA_pairs) == 0:
        return None
    
    # Step 3: Generating descriptions from table
    table_input = linearlize_subtable_fullrow(sample, bridge_entity)
    descriptions = table_to_text.predict_output(table_input)
    descriptions = filter_table_descriptions(descriptions, bridge_entity_name_in_table)
    if len(descriptions) == 0:
        return None

    # Step 4: Blend two parts together
    blended_questions = []
    for table_des in descriptions:
        table_des_no_punct = ' '.join(word_tokenize(table_des)[:-1])
        for text_qa in QA_pairs:
            multi_hop_q = text_qa['question'].replace('@@ENT@@', table_des_no_punct)
            blended_questions.append({'question': multi_hop_q, 'answer': text_qa['answer']})
    
    # down-sampling to speed up
    if len(blended_questions) > q_number:
        blended_questions = random.sample(blended_questions, 3)

    # Step 5: BERT fill in the blank
    for ind, QA in enumerate(blended_questions):
        multi_hop_q = QA['question']
        missing_token = bert_fill_blank.predict(multi_hop_q)
        multi_hop_q = multi_hop_q.replace('____', missing_token)
        blended_questions[ind]['question'] = multi_hop_q
    
    # Step 6: add meta info
    results = []
    for QA in blended_questions:
        info = {
            'question': QA['question'],
            'answer_text': QA['answer'],
            'bridge_entity': bridge_entity_name_in_table, 
            'bridge_entity_loc': bridge_entity_loc, 
            'bridge_entity_text_url': bridge_entity_text_url
        }
        results.append(info)

    return results

'''
Generate a text-to-table question
'''
def Generate_Text_to_Table_Question(sample, q_number = 3):
    # Step 1: get all the cells with link to serve as the answer entities
    cell_with_links = get_bridge_entities(sample)
    
    if len(cell_with_links) == 0:
        return None
    
    # Step 2: randomly sample a few candidate bridge entities and get its "Table passage"
    num_bridge_entities = 20
    bridge_entities = cell_with_links.copy()
    if len(cell_with_links) > num_bridge_entities:
        bridge_entities = random.sample(cell_with_links, num_bridge_entities)
    
    valid_triples = [] # triple of (desc, answer, bridge_entity)
    for bridge_entity in bridge_entities:
        bridge_entity_name = bridge_entity['name']
        bridge_entity_loc = bridge_entity['loc']
        table_input = linearlize_subtable_fullrow(sample, bridge_entity)
        descriptions = table_to_text.predict_output(table_input)
        # get answer (note that bridge entity should not be count)
        answer_entities = get_row_entities(sample['table'], bridge_entity_loc[0], bridge_entity_name)
        if len(answer_entities) == 0:
            continue
        
        # find the descriptions which contain the answer
        matched_pairs = []
        for desc in descriptions:
            if desc.lower().startswith(bridge_entity_name.lower()):
                matched_pairs += match_answer_in_description(desc, answer_entities)

        # add valid triples
        for match in matched_pairs:
            valid_triples.append([match[0], match[1], bridge_entity])

    if len(valid_triples) == 0:
        return None

    # Step 3: generate questions for each valid triple
    valid_questions = []
    for triple in valid_triples:
        table_passage = triple[0]
        answer = triple[1]['name']
        bridge_ent = triple[2]['name']
        QA_pair = qg_nlp.qg_with_answer_text(table_passage, answer)[0]
        # check whether the question is valid
        if QA_pair['answer'] == answer and phrase_match(QA_pair['question'], bridge_ent) >= 0:
            triple[0] = QA_pair['question']
            valid_questions.append(triple)

    if len(valid_questions) == 0:
        return None
    
    # Step 4: Ask question in text for the bridge entity (the bridge entity is the answer)
    valid_results = []
    for question, answer, bridge_entity in valid_questions:
        QA_pairs = ask_question_for_bridge_entity(sample, bridge_entity)
        for qa_pair in QA_pairs:
            text_desc = convert_question_into_desc(qa_pair)
            if not text_desc is None:
                valid_results.append([question, answer, bridge_entity, text_desc])

    # An example of valid_results: 
    # [['What was the time of Cristiano da Matta?', 
    #  {'name': '1:11.691', 'loc': [10, 4, 0]}, 
    #  {'loc': [10, 2, 0], 'name': 'Cristiano da Matta', 'url': '/wiki/Cristiano_da_Matta'}, 
    #  'that won the CART Championship in 2002']]

    # Step 5: Blend two parts together
    blended_questions = []
    for result in valid_results:  
        table_ques = result[0].replace('?', '')
        text_ques = result[3]
        bridge_ent_name = result[2]['name']
        # IMP: blend
        multi_hop_q = table_ques.replace(bridge_ent_name, text_ques, 1) + '?'
        missing_token = bert_fill_blank.predict(multi_hop_q)
        multi_hop_q = multi_hop_q.replace('____', missing_token)

        info = {
            'question': multi_hop_q,
            'answer_text': result[1]['name'],
            'bridge_entity': result[1]['name'], 
            'bridge_entity_loc': result[1]['loc'][:2], 
            'bridge_entity_text_url': result[1]['url']
        }
        blended_questions.append(info)

    return blended_questions


'''
Generate a text-to-table-to-text question
Typical example:
Bridge 1: @@ENT@@: Avocado
Q from Text 1: What family does the @@ENT@@ belong to?, 'answer': 'Lauraceae'
Table desc: Avocado is the first ingredient in @@ENT2@@ burger. (@@ENT2@@: California)
Bridge 2: California
Q from Text 2: the ____ that is the third - largest state by area
Blended: 
What family does the ____ that is the first ingredient in the ____ that is the third - largest state by area burger belong to?'
'''
def Generate_Text_to_Table_to_Text_Question(sample, num_bridge_entities = 20):
    # Step 1: randomly select a bridge entity
    cell_with_links = get_bridge_entities(sample)
    if len(cell_with_links) == 0:
        return None

    entity_names = get_all_entity_names(sample, cell_with_links)
    if len(entity_names) == 0:
        return None

    bridge_entities = cell_with_links.copy()
    if len(cell_with_links) > num_bridge_entities:
        bridge_entities = random.sample(cell_with_links, num_bridge_entities)

    for bridge_entity in bridge_entities:
        bridge_entity_name_in_table = bridge_entity['name']
        bridge_entity_text_url = bridge_entity['url']
        bridge_entity_loc = bridge_entity['loc']

        # Step 2: Asking questions in the text
        bridge1_QA_pairs = ask_questions_in_text(sample, bridge_entity, num_sent=5)
        if len(bridge1_QA_pairs) == 0:
            return None

        # Step 3: Generating descriptions from table
        table_input = linearlize_subtable_fullrow(sample, bridge_entity)
        descriptions = table_to_text.predict_output(table_input)
        descriptions = filter_table_descriptions(descriptions, bridge_entity_name_in_table)
        if len(descriptions) == 0:
            return None

        # Step 4: Find the second bridge entity in the generated table description
        valid_descs = []
        for desc in descriptions:
            for entity in entity_names:
                if entity['url'] == bridge_entity_text_url:
                    continue
                updated_desc = is_desc_contains_entity(desc, entity)
                if not updated_desc is None:
                    valid_descs.append([updated_desc, entity])
                    break
        if len(valid_descs) == 0:
            return None

        # valid_descs: 
        # [['the ____ that won the gold medal in @@ENT2@@.', 
        # {'loc': [0, 2, 0], 'name': 'Alpine skiing', 'url': '/wiki/Alpine_skiing_at_the_1988_Winter_Olympics', 'text_name': 'Alpine Skiing', 'table_name': 'Alpine skiing'}]]

        # Step 5: Ask question in text for the second bridge entity (the bridge entity is the answer)
        valid_results = []
        for desc, bridge_entity in valid_descs:
            QA_pairs = ask_question_for_bridge_entity(sample, bridge_entity)
            for qa_pair in QA_pairs:
                text_desc = convert_question_into_desc(qa_pair)
                if not text_desc is None:
                    valid_results.append([desc, bridge_entity, text_desc]) 
        if len(valid_results) == 0:
            return None

        # Step 6: Blend three parts together
        blended_questions = []
        for table_des, bridge2, text_desc in valid_results:
            table_des_no_punct = table_des[:-1]
            for text_qa in bridge1_QA_pairs:
                multi_hop_q = text_qa['question'].replace('@@ENT@@', table_des_no_punct)
                multi_hop_q = multi_hop_q.replace('@@ENT2@@', text_desc)
                blended_questions.append({'question': multi_hop_q, 'answer': text_qa['answer'], 'bridge2': bridge2})

        # Step 5: BERT fill in the blank
        for ind, QA in enumerate(blended_questions):
            multi_hop_q = QA['question']
            # fill the first blank
            missing_token = bert_fill_blank.predict(multi_hop_q)
            multi_hop_q = multi_hop_q.replace('____', missing_token, 1)
            # fill the second blank
            missing_token = bert_fill_blank.predict(multi_hop_q)
            multi_hop_q = multi_hop_q.replace('____', missing_token)
            blended_questions[ind]['question'] = multi_hop_q
        
        # Step 6: add meta info
        results = []
        for QA in blended_questions:
            info = {
                'question': QA['question'],
                'answer_text': QA['answer'],
                'bridge_entity': [bridge_entity_name_in_table, QA['bridge2']['name']],
                'bridge_entity_loc': [bridge_entity_loc, QA['bridge2']['loc']],
                'bridge_entity_text_url': [bridge_entity_text_url, QA['bridge2']['url']]
            }
            results.append(info)

        return results


'''
Generate a text-only questions (For baseline)
'''
def Generate_Text_Only_Question(sample, q_number = 4):
    # Step 1: randomly select a bridge entity
    cell_with_links = get_bridge_entities(sample)
    
    if len(cell_with_links) == 0:
        return None
    
    bridge_entities = cell_with_links.copy()
    if len(cell_with_links) > q_number:
        bridge_entities = random.sample(cell_with_links, q_number)

    results = []
    for bridge_entity in bridge_entities:
        bridge_entity_name_in_table = bridge_entity['name']
        bridge_entity_text_url = bridge_entity['url']
        bridge_entity_loc = bridge_entity['loc']

        # Step 2: Asking questions in the text
        QA_pairs = ask_questions_in_text(sample, bridge_entity, num_sent = 5, replace_with_ENT = False)
        if len(QA_pairs) == 0:
            continue
        
        selected_QA_pair = random.sample(QA_pairs, 1)[0]

        # Step 6: add meta info
        info = {
            'question': selected_QA_pair['question'],
            'answer_text': selected_QA_pair['answer'],
            'bridge_entity': bridge_entity_name_in_table,
            'bridge_entity_loc': bridge_entity_loc, 
            'bridge_entity_text_url': bridge_entity_text_url
        }
        results.append(info)

    return results


'''
Generate a table-only questions (For baseline)
'''
def Generate_Table_Only_Question(sample, q_number = 4):
    # Step 1: get all the cells with link to serve as the answer entities
    cell_with_links = get_bridge_entities(sample)
    
    if len(cell_with_links) == 0:
        return None
    
    # Step 2: randomly sample a few candidate bridge entities and get its "Table passage"
    bridge_entities = cell_with_links.copy()
    if len(cell_with_links) > q_number:
        bridge_entities = random.sample(cell_with_links, q_number)
    
    results = []
    for bridge_entity in bridge_entities:
        bridge_entity_name = bridge_entity['name']
        bridge_entity_loc = bridge_entity['loc']
        bridge_entity_text_url = bridge_entity['url']
        table_input = linearlize_subtable_fullrow(sample, bridge_entity)
        descriptions = table_to_text.predict_output(table_input)
        if len(descriptions) == 0:
            continue
        
        # Step 3: ask question based on table description
        QA_pairs = qg_nlp.qg_without_answer(descriptions[0])
        if len(QA_pairs) == 0:
            return None

        selected_QA_pair = random.sample(QA_pairs, 1)[0]

        # Step 4: add meta info
        info = {
            'question': selected_QA_pair['question'],
            'answer_text': selected_QA_pair['answer'],
            'bridge_entity': bridge_entity_name,
            'bridge_entity_loc': bridge_entity_loc, 
            'bridge_entity_text_url': bridge_entity_text_url
        }
        results.append(info)
    
    return results