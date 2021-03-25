"""
Perform random reasoning for each reasoning chain. 
"""
from .operations import *
import random
from config import qg_nlp, stanza_nlp, bert_fill_blank
import itertools
from nltk.tokenize import word_tokenize

def ask_questions_in_text(passage, bridge_entities, p_index):
    QA_pairs = qg_nlp.qg_without_answer(passage)
    valid_triples = [] # (question, bridge, answer)
    for qa in QA_pairs:
        bridge = include_bridge_entity(qa['question'], bridge_entities)
        if not bridge is None:
            valid_triples.append([qa['question'], bridge, qa['answer'], p_index])
    return valid_triples

def Generate_Text_to_Text_Question(sample, q_number = 3):
    # Step 1: Get potential bridge entities
    passage1, passage2 = sample[0], sample[1]
    bridge_entities = get_bridge_entites(passage1, passage2)
    if len(bridge_entities) == 0:
        return None

    # Step 2: Get questions in Passage1 & Passage2 which contains at least one bridge entity
    valid_triples = ask_questions_in_text(passage1, bridge_entities, 1)
    valid_triples += ask_questions_in_text(passage2, bridge_entities, 2)
    if len(valid_triples) == 0:
        return None

    # Step 3: Generate question in Passage2 which the bridge entity as the answer
    valid_results = []
    for question, bridge_entity, answer, p_index in valid_triples:
        QA_pairs = []
        if p_index == 1: 
            QA_pairs = qg_nlp.qg_with_answer_text(passage2, answer)
        elif p_index == 2:
            QA_pairs = qg_nlp.qg_with_answer_text(passage1, answer)
        
        for qa_pair in QA_pairs:
            text_desc = convert_question_into_desc(qa_pair)
            if not text_desc is None:
                valid_results.append([question, answer, bridge_entity, text_desc, p_index])
    
    # An example of valid_results: 
    # ['What was the name of the Northern Arizona Suns team that moved to Bakersfield in the D-League in 2006?', 
    # 'Bakersfield Jam', 
    # 'the Northern Arizona Suns', 
    # "the ____ that is the name of Tyrone Ellis ' head coach", 
    # 2]

    # Step 4: Blend two parts together
    blended_questions = []
    for result in valid_results:  
        table_ques = result[0].replace('?', '')
        text_ques = result[3]
        bridge_ent_name = result[2]
        # IMP: blend
        multi_hop_q = table_ques.replace(bridge_ent_name, text_ques, 1) + '?'
        missing_token = bert_fill_blank.predict(multi_hop_q)
        multi_hop_q = multi_hop_q.replace('____', missing_token)

        info = {
            'question': multi_hop_q,
            'answer_text': result[1],
            'bridge_entity': result[2]
        }
        blended_questions.append(info)

    return blended_questions

def Generate_Comparison_Questions(evidences):
    # Step 1: Locate comparative entities
    # Types to consider: 
    # NORP: Nationalities or religious or political groups.
    # ORG: Companies, agencies, institutions, etc.
    # GPE: Countries, cities, states.
    # LOC: Non-GPE locations, mountain ranges, bodies of water.
    # DATE: Absolute or relative dates or periods.
    
    valid_NER_types = ['NORP', 'GPE', 'LOC', 'DATE']
    evidences_with_ents = []
    for source in evidences:
        source_doc = stanza_nlp(source)
        valid_entities = []
        for ent in source_doc.ents:
            if ent.type in valid_NER_types:
                valid_entities.append([ent.text, ent.type])
        
        if len(valid_entities) > 0:
            evidences_with_ents.append({'text': source, 'ents': valid_entities})

    if len(evidences_with_ents) == 0:
        return None

    # Step 2: QG_for_comparative_entities
    evidences_with_ques = []
    for evidence in evidences_with_ents:
        passage = evidence['text']
        questions = []
        for ent in evidence['ents']:
            qa_pair = qg_nlp.qg_with_answer_text(passage, ent[0])
            for ques in qa_pair:
                questions.append({'question': ques['question'], 'answer': ques['answer'], 'type': ent[1]})
        
        if len(questions) > 0:
            evidences_with_ques.append({'text': passage, 'questions': questions})
    
    if len(evidences_with_ques) == 0:
        return None

    # Step 3: generate comparative questions based on template
    valid_questions = []

    for i in range(len(evidences_with_ques)):
        for j in range(i+1, len(evidences_with_ques)):
            eve1, eve2 = evidences_with_ques[i], evidences_with_ques[j]
            # for each question pair
            for ques1 in eve1['questions']:
                for ques2 in eve2['questions']:
                    # birthdate question
                    if ques1['type'] == 'DATE' and ques2['type'] == 'DATE':
                        multi_ques = compose_brithdate_question(ques1, ques2)
                        if multi_ques is not None:
                            valid_questions += multi_ques
                    # liveplace & location question
                    if ques1['type'] in ['GPE', 'LOC'] and ques2['type'] in ['GPE', 'LOC']:
                        multi_ques = compose_liveplace_question(ques1, ques2)
                        if multi_ques is not None:
                            valid_questions += multi_ques
                        multi_ques = compose_location_question(ques1, ques2)
                        if multi_ques is not None:
                            valid_questions += multi_ques
                    # nationality question
                    if ques1['type'] == 'NORP' and ques2['type'] == 'NORP':
                        multi_ques = compose_nationality_question(ques1, ques2)
                        if multi_ques is not None:
                            valid_questions += multi_ques

    if len(valid_questions) == 0:
        return None

    return valid_questions
