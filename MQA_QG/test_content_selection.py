from config import *

from HybridQA.hybridQA_loader import HybridQA_Dataset
from HybridQA.reasoning import *

from HotpotQA.hotpotQA_loader import HotpotQA_Dataset
from HotpotQA.reasoning import *

from tqdm import tqdm
import json
import random

def test_HybridQA(args):
    hybridQA = HybridQA_Dataset(args)
    questions = []

    for i in tqdm(range(50)):
        # sample_table = hybridQA.dataset['376']
        sample_key = random.sample(list(hybridQA.dataset.keys()), 1)[0]
        # ques = Generate_Text_to_Table_Question(sample_table)
        # ques = Generate_Text_to_Table_to_Text_Question(hybridQA.dataset[sample_key])
        # ques = Generate_Text_Only_Question(hybridQA.dataset[sample_key])
        ques = Generate_Table_Only_Question(hybridQA.dataset[sample_key])
        if not ques is None:
            questions += ques

    import ipdb; ipdb.set_trace()

def test_HotpotQA(args):
    hotpotQA = HotpotQA_Dataset(args)
    
    questions = []
    for i in tqdm(range(50)):
        sample_key = random.sample(list(hotpotQA.dataset.keys()), 1)[0]
        # ques = Generate_Text_to_Text_Question(hotpotQA.dataset[sample_key])
        ques = Generate_Comparison_Questions(hotpotQA.dataset[sample_key])
        if not ques is None:
            questions += ques

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    if EXPERIMENT == 'HybridQA':
        test_HybridQA(parser.parse_args())
    elif EXPERIMENT == 'HotpotQA':
        test_HotpotQA(parser.parse_args())
