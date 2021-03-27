"""
Configuration of our project.
"""
import argparse
from Operators import T5_QG
from Operators.Table_to_Text import get_GPT2_Predictor
from Operators.BERT_fill_blank import BERT_fill_blanker
import stanza

#-------------------------------START OF CONFIGURATION--------------------------------------#

###### Global Settings
EXPERIMENT = 'HybridQA' # The experiment you want to run, choose 'HotpotQA' or 'HybridQA'
qg_gp_index = 5  # gpu device to run the QG module
bert_gpu_index = 3 # gpu device to run the BERT module
table_gpu_index = 3 # gpu devide to run the Table2Text module
QUESTION_TYPE = 'table2text' # the type of question you want to generate
# for hybridQA, the options are: 'table2text', 'text2table', 'text_only', 'table_only'
# for hotpotQA, the options are: 'text2text', 'comparison'

###### User-specified data directory
DATA_PATH = '../Data/HybridQA/WikiTables-WithLinks/' # root data directory, '../Data/HybridQA/WikiTables-WithLinks/' for HybridQA; '../Data/HotpotQA/dataset/train.src.txt' for HotpotQA
output_PATH = '../Outputs/train_table_to_text.json' # the json file to store the generated questions
data_range = [0, 20] # for debug use: the range of the dataset you considered (use [0, -1] to use the full dataset)
Table2Text_Model_Path = '../Pretrained_Models/table2text_GPT2_medium_ep9.pt' # the path to the pretrained Table2Text model

# output_PATH, data_range = "", "", []

# if EXPERIMENT == 'HybridQA':
#     DATA_PATH = '/mnt/edward/data/liangming/Projects/HybridQA/WikiTables-WithLinks/'
#     output_PATH = '../HybridQA/HybridQA/synthesized_data/train_Table_Only_Part3.json'
#     data_range = [0, 10000]
# elif EXPERIMENT == 'HotpotQA': # /mnt/edward
#     DATA_PATH = '/data/liangming/Projects/SpanBERT/data/dataset/train.src.txt'
#     output_PATH = '/data/liangming/Projects/SpanBERT/data/dataset/train.qa.comparison_part9.txt'
#     data_range = [90000, -1]
# else:
#     raise(NotImplementedError)
 
#-------------------------------END OF CONFIGURATION--------------------------------------#

# QG NLP object
print('Loading QG module >>>>>>>>')
qg_nlp = T5_QG.pipeline("question-generation", model='valhalla/t5-base-qg-hl', qg_format="highlight", gpu_index = qg_gp_index)
print('QG module loaded.')

table_to_text = None
if EXPERIMENT == 'HybridQA':
    # Table-to-Text object
    print('Loading Table-to-Text module >>>>>>>>')
    table_to_text = get_GPT2_Predictor(
        Table2Text_Model_Path,
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
        default=DATA_PATH + "tables_tok/",
        type=str, help='path of the hybridQA tables')

    parser.add_argument(
        '--text_dir',
        default=DATA_PATH + "request_tok/",
        type=str, help='path of the hybridQA texts')
