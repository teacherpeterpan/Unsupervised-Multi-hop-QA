'''
Rank the sythesized questions based on their PPL under GPT2-Medium
'''
from tqdm import tqdm
import json
import math, torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

print('Loading GPT2 model......')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

def PPL_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, labels=tensor_input)
    return math.exp(loss[0].item())

def load_all_questions(datapath):
    with open(datapath, 'r') as f:
        dataset = json.load(f)
    questions = {index: sample['question'] for index, sample in enumerate(dataset)}
    return questions

def rank_PPL_score(corpus):
    scores = []
    corpus = list(corpus.items())
    for ques_id, ques in tqdm(corpus):
        scores.append([ques_id, PPL_score(ques), ques])
    
    sorted_scores = sorted(scores, key = lambda a:a[1])
    return sorted_scores

# output: {'question_id': [LM_score, question]}
def obtain_score_dict(input_file, output_file):
    corpus = load_all_questions(input_file)
    sorted_scores = rank_PPL_score(corpus)

    with open(output_file, 'w') as f:
        f.write(json.dumps(sorted_scores, indent=2))

if __name__ == "__main__":
    input_file = '/mnt/edward/data/liangming/Projects/HybridQA/HybridQA/synthesized_data/train_R2.json'
    output_file = '/mnt/edward/data/liangming/Projects/HybridQA/HybridQA/synthesized_data/R2_ranking.json'
    obtain_score_dict(input_file, output_file)