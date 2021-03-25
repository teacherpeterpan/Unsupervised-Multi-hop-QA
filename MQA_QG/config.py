"""
Configuration of our project.
"""
import argparse
from QuestionRealization import T5_QG
from QuestionRealization.Table_to_Text import get_GPT2_Predictor
from QuestionRealization.BERT_fill_blank import BERT_fill_blanker
import stanza

###### Global Settings
EXPERIMENT = 'HotpotQA' # Or HybridQA 
qg_gp_index = 5
bert_gpu_index = 3
table_gpu_index = 3

###### User-specified data directory
DATA_PATH, output_PATH, data_range = "", "", []

if EXPERIMENT == 'HybridQA':
    DATA_PATH = '/mnt/edward/data/liangming/Projects/HybridQA/WikiTables-WithLinks/'
    output_PATH = '../HybridQA/HybridQA/synthesized_data/train_Table_Only_Part3.json'
    data_range = [0, 10000]
elif EXPERIMENT == 'HotpotQA': # /mnt/edward
    DATA_PATH = '/data/liangming/Projects/SpanBERT/data/dataset/train.src.txt'
    output_PATH = '/data/liangming/Projects/SpanBERT/data/dataset/train.qa.comparison_part9.txt'
    data_range = [90000, -1]
else:
    raise(NotImplementedError)
 
#---------------------------------------------------------------------#

# QG NLP object
print('Loading QG module >>>>>>>>')
qg_nlp = T5_QG.pipeline("question-generation", model='valhalla/t5-base-qg-hl', qg_format="highlight", gpu_index = qg_gp_index)
print('QG module loaded.')

table_to_text = None
if EXPERIMENT == 'HybridQA':
    # Table-to-Text object
    print('Loading Table-to-Text module >>>>>>>>')
    table_to_text = get_GPT2_Predictor(
        '/mnt/edward/data/liangming/Projects/Table2Text/GPT-Table-to-Seq-ToTTo/models/GPT_medium_ep9.pt',
        num_samples = 1, 
        gpu_index = table_gpu_index)
    print('Table-to-Text module loaded.')

# Load BERT_fill_blanker
print('Loading BERT blender >>>>>>>>')
bert_fill_blank = BERT_fill_blanker(gpu_index = bert_gpu_index)
print('BERT blender loaded.')

# Stanza NLP object
# stanza.download('en')
stanza_nlp = stanza.Pipeline('en', use_gpu = True)

# parser used to read argument
parser = argparse.ArgumentParser(description='HibridQG')

# input files
parser.add_argument(
    '--data_path',
    default=DATA_PATH,
    type=str, help='path of the hybridQA dataset')

if EXPERIMENT == 'HybridQA':
    parser.add_argument(
        '--table_dir',
        default=DATA_PATH + "tables/",
        type=str, help='path of the hybridQA tables')

    parser.add_argument(
        '--text_dir',
        default=DATA_PATH + "request/",
        type=str, help='path of the hybridQA texts')
