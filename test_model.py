import os
import pandas as pd
import pickle
from LM import LM_normal
import numpy as np
import json
import argparse
from utils import *
import logging
import math

parser = argparse.ArgumentParser(description='Test model on the generated data')
parser.add_argument('-DATA', type=str, default='./data/deductive_reasoning_data_test.json')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/test_results/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/')
parser.add_argument('-bf16', action='store_true')
parser.add_argument('-prompt1', action='store_true')
parser.add_argument('-test_batch_size', type=int, default=1)
parser.add_argument('-test_all_order', action='store_true')
parser.add_argument('-num_premises', type=int, default=3)
parser.add_argument('-suffix', type=str, default='original')
parser.add_argument('-device', type=str, default='cpu')

args = parser.parse_args()

device = args.device

suffix = args.model_type + '_' + args.suffix
if args.prompt1:
    suffix += '_prompt1'
if args.test_all_order:
    suffix += '_testallorder'
if args.num_premises != 3:
    suffix += '_premises' + str(args.num_premises)
RESULTS_DIR = os.path.join(args.RESULTS_DIR, suffix)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

dataset = load_from_json(args.DATA)

model_path = args.model_path
if args.model_type == 'llama2':
    model = LM_normal(model_path, device, parse_model_type='llama2', bf16=args.bf16)
    max_new_tokens = 2
else:
    model = LM_normal(model_path, device, bf16=args.bf16)
    max_new_tokens = 1

if args.num_premises == 3:
    str_premises = 'three'
elif args.num_premises == 4:
    str_premises = 'four'
elif args.num_premises == 5:
    str_premises = 'five'
elif args.num_premises == 6:
    str_premises = 'six'
    
prompt_syllogisms = "You are an expert in performing logical reasoning. I will present {} premises and one conclusion each round. The premises describe a series of relationships among monosyllabic pseudowords and adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response.".format(str_premises)
prompt_transitive = "You are an expert in performing logical reasoning. I will present {} premises and one conclusion each round. The premises describe relationships among imaginary characters with comparative adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response.".format(str_premises)
prompt1 = "{} premises and one conclusion will be presented. Judge if the conclusion can be drawn from the premises logically. Only answer 'True' or 'False' at the beginning of the response.\n\n".format(str_premises)

path_results = os.path.join(RESULTS_DIR, f'behaviour_results.pkl')
if not args.test_all_order:
    num_order = 1
else:
    num_order = math.factorial(args.num_premises)
if os.path.exists(path_results):
    with open(path_results, "rb") as f:
        correct_all = pickle.load(f)
    acc = average_list(correct_all)
    assert len(correct_all) == (len(dataset['syllogisms']) + len(dataset['transitive'])) * num_order
    syll_acc = average_list(correct_all[:len(dataset['syllogisms'])*num_order])
    tran_acc = average_list(correct_all[len(dataset['syllogisms'])*num_order:])
else:
    if args.prompt1:
        if args.test_batch_size == 1:
            acc, syll_acc, tran_acc, correct_all = test_model(model, dataset, prompt1, prompt1, max_new_tokens, all_order=args.test_all_order, num_premises=args.num_premises, return_results=True)
        else:
            acc, syll_acc, tran_acc, correct_all = test_model_batch(model, dataset, prompt1, prompt1, max_new_tokens, args.test_batch_size, all_order=args.test_all_order, num_premises=args.num_premises, return_results=True)
    else:
        if args.test_batch_size == 1:
            acc, syll_acc, tran_acc, correct_all = test_model(model, dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, all_order=args.test_all_order, num_premises=args.num_premises, return_results=True)
        else:
            acc, syll_acc, tran_acc, correct_all = test_model_batch(model, dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, args.test_batch_size, all_order=args.test_all_order, num_premises=args.num_premises, return_results=True)
    
    with open(path_results, "wb") as f:
        pickle.dump(correct_all, f)


log_path = os.path.join(RESULTS_DIR, 'test.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print(suffix)
print('Overall acc is: {:.3f}, syllogisms acc is: {:.3f}, transitive acc is: {:.3f}'.format(acc, syll_acc, tran_acc))
logging.info('Overall acc is: {:.3f}, syllogisms acc is: {:.3f}, transitive acc is: {:.3f}'.format(acc, syll_acc, tran_acc))

# analyze each question type
if args.num_premises == 3:
    categories = [
        "2_true_affirm", "2_false_affirm",
        "3_true_affirm", "3_false_affirm",
        "2_true_negate", "2_false_negate",
        "3_true_negate", "3_false_negate"
    ]
    trial_num_per_type = 100
elif args.num_premises == 4:
    categories = [
        "4_true_affirm", "4_false_affirm",
        "4_true_negate", "4_false_negate"
    ]
    trial_num_per_type = 100
elif args.num_premises == 5:
    categories = [
        "5_true_affirm", "5_false_affirm",
        "5_true_negate", "5_false_negate"
    ]
    trial_num_per_type = 20
elif args.num_premises == 6:
    categories = [
        "6_true_affirm", "6_false_affirm",
        "6_true_negate", "6_false_negate"
    ]
    trial_num_per_type = 10

print('Syllogisms:')
logging.info('Syllogisms:')
for i in range(len(categories)):
    idx_start = i * trial_num_per_type * num_order
    idx_end = (i + 1) * trial_num_per_type * num_order
    print(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
    logging.info(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
print('Transitive:')
logging.info('Transitive:')
for i in range(len(categories)):
    idx_start = (i + len(categories)) * trial_num_per_type * num_order
    idx_end = (i + 1 + len(categories)) * trial_num_per_type * num_order
    print(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
    logging.info(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
