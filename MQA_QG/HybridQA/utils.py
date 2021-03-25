"""
Utility functions for content selection 
"""

from config import qg_nlp, stanza_nlp

# return the first identified named entity and its type from text
def get_first_NER(source):
    source_doc = stanza_nlp(source)
    if len(source_doc.ents) > 0:
        first_ent = source_doc.ents[0]
        return [first_ent.text, 'ENTITY_' + first_ent.type]
    return None, None

def phrase_overlap(P1, P2):
    L1 = P1.split()
    L2 = P2.split()
    return len(set(L1) & set(L2)) > 0

# given a question, extract the entity and relation (based on grammar rules)
def extract_key_struc_from_question(source):
    source_doc = stanza_nlp(source)
    sent = source_doc.sentences[0]
    head_index, verb_index = -1, -1
    # find head word (head_word -> (nsubj) -> verb)
    for ind, word in enumerate(sent.words):
        if word.deprel.startswith('nsubj') and word.xpos.startswith('NN'):
            head_index = ind
            verb_index = word.head
            break
    if head_index == -1 or verb_index==-1:
        return None, None
    # get the entity phrase
    end_pos = head_index
    for ind in range(head_index, len(sent.words)):
        if not sent.words[ind].xpos.startswith('NN'):
            end_pos = ind
            break
    start_index = head_index
    while sent.words[start_index].xpos.startswith('NN') and start_index >=0:
        start_index -= 1
    # start_index = head_index if start_index == head_index else start_index + 1
    entity_phrase = [word.text for word in sent.words[start_index + 1: end_pos]]
    verb_phrase = [word.text for word in sent.words[verb_index - 1: -1]]
    return ' '.join(entity_phrase), ' '.join(verb_phrase)

# Rule: What is the racing division of Ferrari? => that is the racing division of Ferrari
def get_predicate(source):
    source_doc = stanza_nlp(source)
    sent = source_doc.sentences[0]

    if sent.words[-1].text == '?' and sent.words[0].xpos.startswith('W'):
        return 'the ____ that ' + ' '.join([w.text for w in sent.words[1:-1]])
    
    return None

# return all the entities in a certain row
def get_row_entities(table, row_index, bridge_entity_name):
    row_entities = []
    for c_ind, col in enumerate(table['data'][row_index]):
        for e_ind, name in enumerate(col[0]):
            if not name == bridge_entity_name:
                row_entities.append({'name': name, 'loc': [row_index, c_ind, e_ind], 'url': table['data'][row_index][c_ind][1][e_ind]})
    return row_entities

def phrase_match(long_str, short_str):
    long_str = ' ' + long_str[:-1] + ' '
    short_str = ' ' + short_str + ' '
    return long_str.lower().find(short_str.lower())

# given a description, to match whether the description contains at least one answer in the answer set
def match_answer_in_description(desc, ans_set):
    matched_pairs = []
    for ans in ans_set:
        if phrase_match(desc, ans['name']) >= 0:
            matched_pairs.append([desc, ans])
    return matched_pairs




