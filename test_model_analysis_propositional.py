import os
import pandas as pd
import pickle
from LM import LM_normal
import numpy as np
import json
import argparse
from utils import *
import logging

parser = argparse.ArgumentParser(description='Test model on the generated data')
parser.add_argument('-DATA', type=str, default='./data/deductive_reasoning_data_test_propositional.json')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/test_results_propositional/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data3/mqxiao/LLM/Qwen/Qwen2-1.5B-Instruct/')
parser.add_argument('-bf16', action='store_true')
parser.add_argument('-device', type=str, default='cpu')
parser.add_argument('-log_stdout', action='store_true', help='Also print logs to stdout')

args = parser.parse_args()

device = args.device

RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

dataset = load_from_json(args.DATA)

path_results = os.path.join(RESULTS_DIR, f'behaviour_results.pkl')
    
if os.path.exists(path_results):
    print('loading from ' + path_results)
    with open(path_results, "rb") as f:
        [model_ans_all, label_all, correct_all] = pickle.load(f)
else:
    log_path = os.path.join(RESULTS_DIR, 'test.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.log_stdout:
        logging.getLogger().addHandler(logging.StreamHandler())
    
    model_path = args.model_path
    if 'llama3' in args.model_type:
        model = LM_normal(model_path, device, parse_model_type='llama3', bf16=args.bf16)
    elif args.model_type == 'llama2':
        model = LM_normal(model_path, device, parse_model_type='llama2', bf16=args.bf16)
    elif 'qwen' in args.model_type:
        model = LM_normal(model_path, device, parse_model_type='qwen', bf16=args.bf16)
    else:
        model = LM_normal(model_path, device, bf16=args.bf16)
    max_new_tokens = 1
    if args.model_type == 'llama2':
        max_new_tokens = 2
    
    prompt_propositional = "You are an expert in performing logical reasoning. I will present two premises and one conclusion each round. The premises describe logical arguments. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."

    model_ans_all = []
    label_all = []
    correct_all = []
    
    # propositional
    for data in dataset['propositional']:
        prompt = prompt_propositional
        messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\nConclusion:\n%s." % (data['premise1'].strip(), data['premise2'].strip(), data['conclusion'].strip())},
        ]
    
        ans = model(messages, max_new_tokens=max_new_tokens)
        logging.info(data['trial_type'] + ', ' + ans)
        if ans in ['True', 'true', 'True.']:
            model_ans = 1
        elif ans in ['False', 'false', 'False.']:
            model_ans = 0
        else:
            model_ans = None
            logging.info('----------')
            logging.info('No answer. The output is ' + ans)
            logging.info('----------')
        model_ans_all.append(model_ans)
    
        trial_type = data['trial_type']
        if 'true' in trial_type:
            label = 1
        else:
            label = 0
        label_all.append(label)
        correct_all.append(model_ans==label)
    logging.info('Propositional done')
    
    logging.info('--------------------')
    logging.info('Overall acc is: {:.3f}'.format(average_list(correct_all)))
    logging.info('--------------------')
    
    path_results = os.path.join(RESULTS_DIR, f'behaviour_results.pkl')
        
    with open(path_results, "wb") as f:
        pickle.dump([model_ans_all, label_all, correct_all], f)

print('Overall acc is: {:.3f}'.format(average_list(correct_all)))
categories = [
    "modus_ponens_true", "modus_ponens_false",
    "modus_tollens_true", "modus_tollens_false",
    "disjunction_elimination_true", "disjunction_elimination_false",
]
trial_num_per_type = 100
for i in range(len(categories)):
    idx_start = i * trial_num_per_type
    idx_end = (i + 1) * trial_num_per_type
    #print(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
    logging.info(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
