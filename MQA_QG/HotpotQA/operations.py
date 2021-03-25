"""
Define atomic operations in HotpotQA. 
"""
from config import stanza_nlp
import dateparser

def phrase_match(long_str, short_str):
    long_str = ' ' + long_str[:-1] + ' '
    short_str = ' ' + short_str + ' '
    return long_str.find(short_str)

'''
Find potential bridge entities. 
1) Find entities for text1 (NE1) and text2 (NE2)
2) If NE1 & NE2 >0, those are selected as bridge entities
3) Otherwise, if an entity in NE1 appears as a none-entity in text2, we also treat it as a bridge entity
4) Same for NE2 -> text1 
'''
def get_bridge_entites(passage1, passage2):
    doc1 = stanza_nlp(passage1)
    doc2 = stanza_nlp(passage2)
    entity_set1 = set([ent.text for ent in doc1.ents])
    entity_set2 = set([ent.text for ent in doc2.ents])

    overlap_entities = entity_set1 & entity_set2
    if len(overlap_entities) > 0:
        return list(overlap_entities)
    else:
        bridge_entities = []
        for ent in entity_set1:
            if phrase_match(passage2, ent) >= 0:
                bridge_entities.append(ent)
        for ent in entity_set2:
            if phrase_match(passage1, ent) >= 0:
                bridge_entities.append(ent)
    return []

'''
Judge if at least one bridge entity appears in the question
'''
def include_bridge_entity(question, bridge_entities):
    for ent in bridge_entities:
        if phrase_match(question, ent) >= 0:
            return ent
    return None

'''
Convert a text question into a description
'''
def convert_question_into_desc(qa_pair):
    predicate = get_predicate(qa_pair['question'])    
    return predicate

# Rule: What is the racing division of Ferrari? => that is the racing division of Ferrari
def get_predicate(source):
    source_doc = stanza_nlp(source)
    sent = source_doc.sentences[0]

    if sent.words[-1].text == '?' and sent.words[0].xpos.startswith('W'):
        return 'the ____ that ' + ' '.join([w.text for w in sent.words[1:-1]])
    
    return None

def get_person_entity(ents):
    for ent in ents:
        if ent.type == 'PERSON':
            return ent.text
    return None

def compose_brithdate_question(ques1, ques2):
    if ques1['question'].find('born') < 0 or ques2['question'].find('born') < 0:
        return None
    
    ques1_doc = stanza_nlp(ques1['question'])
    ques2_doc = stanza_nlp(ques2['question'])
    if len(ques1_doc.ents) == 0 or len(ques2_doc.ents) == 0:
        return None
    
    ent1 = get_person_entity(ques1_doc.ents)
    ent2 = get_person_entity(ques2_doc.ents)
    if ent1 is None or ent2 is None:
        return None
    
    multiQ = f'Who was born first, {ent1} or {ent2}?'
    date1 = dateparser.parse(ques1['answer'])
    date2 = dateparser.parse(ques2['answer'])
    if date1 is None or date2 is None:
        return None
    answer = ent1 if date1 < date2 else ent2
    return [{'question': multiQ, 'answer_text': answer}]

def get_first_entity(ents):
    for ent in ents:
        return ent.text
    return None

def compose_location_question(ques1, ques2):
    if ques1['question'].find('located') < 0 or ques2['question'].find('located') < 0:
        return None
    
    ques1_doc = stanza_nlp(ques1['question'])
    ques2_doc = stanza_nlp(ques2['question'])
    if len(ques1_doc.ents) == 0 or len(ques2_doc.ents) == 0:
        return None
    
    ent1 = get_first_entity(ques1_doc.ents)
    ent2 = get_first_entity(ques2_doc.ents)
    if ent1 is None or ent2 is None:
        return None
    
    results = []
    multiQ1 = f'Are {ent1} and {ent2} located in the same place?'
    answer1 = 'yes' if ques1['answer'].lower() == ques2['answer'].lower() else 'no'
    results.append({'question': multiQ1, 'answer_text': answer1})
    
    multiQ2 = f'''Which one is located in {ques1['answer']}, {ent1} or {ent2}?'''
    answer2 = ent1
    results.append({'question': multiQ2, 'answer_text': answer2})
    
    multiQ3 = f'''Which one is located in {ques2['answer']}, {ent1} or {ent2}?'''
    answer3 = ent2
    results.append({'question': multiQ3, 'answer_text': answer3})
    
    multiQ4 = f'''Are both {ent1} and {ent2} located in {ques1['answer']}?'''
    answer4 = 'yes' if ques1['answer'].lower() == ques2['answer'].lower() else 'no'
    results.append({'question': multiQ4, 'answer_text': answer4})
    
    return results

def compose_nationality_question(ques1, ques2):
    if ques1['question'].find('nationality') < 0 or ques2['question'].find('nationality') < 0:
        return None
    
    ques1_doc = stanza_nlp(ques1['question'])
    ques2_doc = stanza_nlp(ques2['question'])
    if len(ques1_doc.ents) == 0 or len(ques2_doc.ents) == 0:
        return None
    
    ent1 = get_person_entity(ques1_doc.ents)
    ent2 = get_person_entity(ques2_doc.ents)
    if ent1 is None or ent2 is None:
        return None
    
    results = []
    multiQ1 = f'Are {ent1} and {ent2} of the same nationality?'
    answer1 = 'yes' if ques1['answer'].lower() == ques2['answer'].lower() else 'no'
    results.append({'question': multiQ1, 'answer_text': answer1})
    
    multiQ2 = f'''Which person is from {ques1['answer']}, {ent1} or {ent2}?'''
    answer2 = ent1
    results.append({'question': multiQ2, 'answer_text': answer2})
    
    multiQ3 = f'''Which person is from {ques2['answer']}, {ent1} or {ent2}?'''
    answer3 = ent2
    results.append({'question': multiQ3, 'answer_text': answer3})
    
    return results

def compose_liveplace_question(ques1, ques2):
    if ques1['question'].find('live') < 0 or ques2['question'].find('live') < 0:
        return None
    
    ques1_doc = stanza_nlp(ques1['question'])
    ques2_doc = stanza_nlp(ques2['question'])
    if len(ques1_doc.ents) == 0 or len(ques2_doc.ents) == 0:
        return None
    
    ent1 = get_person_entity(ques1_doc.ents)
    ent2 = get_person_entity(ques2_doc.ents)
    if ent1 is None or ent2 is None:
        return None
    
    results = []
    multiQ1 = f'Are {ent1} and {ent2} living in the same place?'
    answer1 = 'yes' if ques1['answer'].lower() == ques2['answer'].lower() else 'no'
    results.append({'question': multiQ1, 'answer_text': answer1})
    
    multiQ2 = f'''Which person lives in {ques1['answer']}, {ent1} or {ent2}?'''
    answer2 = ent1
    results.append({'question': multiQ2, 'answer_text': answer2})
    
    multiQ3 = f'''Which person lives in {ques2['answer']}, {ent1} or {ent2}?'''
    answer3 = ent2
    results.append({'question': multiQ3, 'answer_text': answer3})
    
    return results