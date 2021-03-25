'''
Given a linearlized table, predict its description
'''
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer
from .utils import sample_sequence
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# from allennlp.predictors import Predictor

# class Seq2Seq_Table2Text(object):
#     def __init__(self, model_path):
#         self.predictor = Predictor.from_path(model_path)

#     def predict_output(self, table_input):
#         results = self.predictor.predict(table_input)
#         result_list = []
#         for ele in results['predicted_tokens']:
#             result_list.append(' '.join(ele))
#         return result_list

class GPT2_Table2Text(object):
    def __init__(self, args):
        self.device = args['device']
        self.set_seed(args)
        ### Load pretrained model...
        self.tokenizer = GPT2Tokenizer.from_pretrained(args['model'])
        model = GPT2LMHeadModel.from_pretrained(args['model'])
        # model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args['load_from']))
        model.eval()
        self.model = model.to(args['device'])
        self.args = args

    def set_seed(self, args):
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        if args['n_gpu'] > 0:
            torch.cuda.manual_seed_all(args['seed'])

    def parse_data(self, input_table):
        descs = []
        source = self.tokenizer.tokenize(input_table)
        descs.append(self.tokenizer.convert_tokens_to_ids(source))

        length = max([len(_) for _ in descs]) + 1
        for i in range(len(descs)):
            descs[i] = (length - len(descs[i])) * [self.tokenizer.eos_token_id] + descs[i]
        descs = torch.LongTensor(descs)
        return descs

    def predict_output(self, input_table):
        results = []
        with torch.no_grad():
            input_tensor = self.parse_data(input_table).to(self.device)
            input_tensor = input_tensor.repeat(self.args['num_samples'], 1)
            samples = sample_sequence(self.model, 30, input_tensor, [], top_k=self.args['num_samples'], device=self.args['device'])
            samples = samples[:, input_tensor.shape[1]:]
            samples = samples.cpu().data.numpy()

            for s in samples:
                text = self.tokenizer.decode(s, clean_up_tokenization_spaces=True)
                text = text[: text.find(self.tokenizer.eos_token)]
                results.append(text)

        return results

def get_GPT2_Predictor(model_path, num_samples = 3, gpu_index = 0):
    config = {
        'model' : 'gpt2-medium',
        'num_samples': num_samples, 
        'seed': 42,
        'load_from': model_path, 
        'max_len': 800, 
        'device': torch.device("cuda:{}".format(gpu_index)), 
        'n_gpu': 1
    }
    predictor = GPT2_Table2Text(config)
    return predictor

if __name__ == "__main__":
    predictor = get_GPT2_Predictor('../../Pretrained_Models/table2text_GPT2_medium_ep9.pt')
    test_input = '''The table title is 2004 United States Grand Prix . The Pos is 4 . The Driver is Jenson Button . Start describing Jenson Button : '''
    results = predictor.predict_output(test_input)
    print(results)



