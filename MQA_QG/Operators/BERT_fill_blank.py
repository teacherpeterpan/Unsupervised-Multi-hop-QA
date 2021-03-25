import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import nltk

class BERT_fill_blanker(object):
    def __init__(self, gpu_index = 0):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-cased')
        self.device = torch.device('cuda:{}'.format(gpu_index))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentence_orig):
        if '____' not in sentence_orig:
            return sentence_orig

        sentence = sentence_orig.replace('____', '[MASK]')
        
        tokenized_text = self.tokenizer.tokenize(sentence)

        masked_index = tokenized_text.index('[MASK]')
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]

        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
        return predicted_token

