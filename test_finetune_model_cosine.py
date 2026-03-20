import os
import pandas as pd
import pickle
import numpy as np
import torch
import random
from LM_finetune import ModelwithAttentionSupervision
from utils import *
from torch.cuda.amp import autocast, GradScaler
import argparse
import re
import logging

def get_LLM_rep(model, model_type, task_items, prompt_syllogisms, prompt_transitive):
    rep_all_dict = None
    for task_run in sorted(task_items.keys()):
        df = task_items[task_run]
        if 'Transitive' in task_run:
            prompt = prompt_transitive
        else:
            prompt = prompt_syllogisms
    
        for i in range(len(df)):
            messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                    {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (df['premise1'][i].strip(), df['premise2'][i].strip(), df['premise3'][i].strip(), df['conclusion'][i].strip())},
            ]
    
            if 'Transitive' in task_run and '01' in task_run and (i == 7 or i == 13):
                continue
            else:
                # inputs
                x = messages
                if model_type == 'llama2':
                    text = model.tokenizer.apply_chat_template(x, tokenize=False)
                    text += ' '
                    encodeds = model.tokenizer([text], return_tensors='pt').to(device)
                    input_ids = encodeds.input_ids
                else:
                    encodeds = model.tokenizer.apply_chat_template(x, return_tensors='pt', add_generation_prompt=True)
                    input_ids = encodeds.to(device)
                rep_dict = model.get_attention_state(input_ids)
                if rep_all_dict is None:
                    rep_all_dict = {}
                    for k in rep_dict.keys():
                        rep_all_dict[k] = rep_dict[k].unsqueeze(0).detach().cpu().numpy()
                else:
                    for k in rep_all_dict.keys():
                        rep_all_dict[k] = np.concatenate([rep_all_dict[k], rep_dict[k].unsqueeze(0).detach().cpu().numpy()], axis=0)
    return rep_all_dict

def calculate_average_score(sub_list, layer_index_all, LLM_rep, fmri_all_sub, all_W_syll, all_W_tran, q_type='Syllogisms'):
    all_score = np.zeros((len(sub_list), len(layer_index_all)))
    for sub_ind, sub in enumerate(sub_list):
        if q_type == 'Syllogisms':
            q_index = [i for i in range(36)]
        else:
            q_index = [i for i in range(36, 70)]

        for layer in layer_index_all:
            rep = LLM_rep[layer][q_index]
            fmri = fmri_all_sub[sub_ind][q_index]
            if q_type == 'Syllogisms':
                W = all_W_syll[sub][layer]
            else:
                W = all_W_tran[sub][layer]
            score = np.median(calculate_corr_withW(rep, fmri, W))
            all_score[sub_ind][layer - layer_index_all[0]] = score
    avg_score = np.median(np.median(all_score, axis=0))
    return avg_score

def calculate_average_cosine(sub_list, layer_index_all, LLM_rep, fmri_all_sub, all_W_syll, all_W_tran, q_type='Syllogisms'):
    all_score = np.zeros((len(sub_list), len(layer_index_all)))
    for sub_ind, sub in enumerate(sub_list):
        if q_type == 'Syllogisms':
            q_index = [i for i in range(36)]
        else:
            q_index = [i for i in range(36, 70)]

        for layer in layer_index_all:
            rep = LLM_rep[layer][q_index]
            fmri = fmri_all_sub[sub_ind][q_index]
            if q_type == 'Syllogisms':
                W = all_W_syll[sub][layer]
            else:
                W = all_W_tran[sub][layer]
            cosine = calculate_cosine_withW(rep, fmri, W)
            all_score[sub_ind][layer - layer_index_all[0]] = np.mean(cosine)
    avg_cosine = np.median(np.median(all_score, axis=0))
    return avg_cosine


_seed_ = 0
random.seed(_seed_)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(_seed_)


parser = argparse.ArgumentParser(description='Test finetuned model')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/test_finetune_score/')
parser.add_argument('-LLM_PREV_RESULTS_DIR', type=str, default='./results/activations_results/')
parser.add_argument('-FMRI_RESULTS_DIR', type=str, default='./fmri_data/preprocessed_data_glmsinglesep_newdrroi_topksep/top-10%/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/')

parser.add_argument('-add_intercept', action='store_true') # not implemented
parser.add_argument('-analyze_type', type=str, default='deductive_reasoning')
parser.add_argument('-loss_type', type=str, default='cosine')
parser.add_argument('-ridge_alpha', type=float, default=100.)
parser.add_argument('-use_ridgecv', action='store_true')

# suffix for log directory
parser.add_argument('-suffix', type=str, default='')

# device
parser.add_argument('-device', type=str, default='cpu')
parser.add_argument('-log_stdout', action='store_true', help='Also print logs to stdout')

args = parser.parse_args()

DATA_DIR = args.DATA_DIR

RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type, args.suffix)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

log_path = os.path.join(RESULTS_DIR, 'results.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if args.log_stdout:
    logging.getLogger().addHandler(logging.StreamHandler())
    
device = args.device

max_new_tokens = 1
parse_model_type = 'default'
model_path = args.model_path
if args.model_type == 'llama2':
    parse_model_type = 'llama2'
    layer_num = 32
    max_new_tokens = 2
elif args.model_type == 'mistral':
    layer_num = 32
elif args.model_type == 'qwen1-5b':
    layer_num = 28
elif args.model_type == 'qwen7b':
    layer_num = 28
elif args.model_type == 'llama3':
    layer_num = 32
elif args.model_type == 'qwen72b':
    layer_num = 80
elif args.model_type == 'phi4-mini':
    layer_num = 32
elif args.model_type == 'gemma2_9b':
    layer_num = 42
elif args.model_type == 'qwen3_4b':
    layer_num = 36
elif args.model_type == 'llama3-3_70b':
    layer_num = 80
else:
    raise("Error! Unsupported model type")

layer_index_all = list(range(layer_num//4, layer_num*3//4))

logging.info('layer index')
logging.info(layer_index_all)

model = ModelwithAttentionSupervision(model_path, layer_index_all, device, False, False, False, args.model_type)

LLM_PREV_RESULTS_DIR = os.path.join(args.LLM_PREV_RESULTS_DIR, args.model_type)
FMRI_RESULTS_DIR = os.path.join(args.FMRI_RESULTS_DIR, 'all_extracted_beta_' + args.analyze_type)

filtered_index = ['sub-1010', 'sub-1016', 'sub-1026', 'sub-1031', 'sub-1032', 'sub-1035', 'sub-1021']

task_items, data_dict, sub_list, model_ans_all, label_all, correct_all = load_and_filter_behavior_results(DATA_DIR, LLM_PREV_RESULTS_DIR, filtered_index)
task_run_list = sorted(task_items.keys())
# get behavior results
model_acc, sub_task_acc = get_model_human_behavior(correct_all, data_dict, sub_list)
# load prev LLM features
LLM_rep_all = load_LLM_features(LLM_PREV_RESULTS_DIR, task_run_list)
# load all fmri data
fmri_all_sub, _ = get_all_fmri_data_latest(sub_list, FMRI_RESULTS_DIR)

# model prompt
prompt_syllogisms = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe a series of relationships among monosyllabic pseudowords and adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."
prompt_transitive = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe relationships among imaginary characters with comparative adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."

# get representation direction
all_W_syll = {}
all_W_tran = {}
if args.add_intercept:
    all_b_syll = {}
    all_b_tran = {}
for sub_ind, sub in enumerate(sub_list):
    all_W_syll[sub] = {}
    all_W_tran[sub] = {}
    if args.add_intercept:
        all_b_syll[sub] = {}
        all_b_tran[sub] = {}

    for layer in layer_index_all:
        if args.add_intercept:
            W_syll, b_syll = get_W_info_cv(LLM_rep_all[:36], fmri_all_sub[sub_ind, :36], layer, None, return_b=True)
            W_tran, b_tran = get_W_info_cv(LLM_rep_all[36:], fmri_all_sub[sub_ind, 36:], layer, None, return_b=True)
        else:
            W_syll = get_W_info_cv(LLM_rep_all[:36], fmri_all_sub[sub_ind, :36], layer, None)
            W_tran = get_W_info_cv(LLM_rep_all[36:], fmri_all_sub[sub_ind, 36:], layer, None)

        all_W_syll[sub][layer] = W_syll
        all_W_tran[sub][layer] = W_tran
        if args.add_intercept:
            all_b_syll[sub][layer] = b_syll
            all_b_tran[sub][layer] = b_tran


# test scores
LLM_rep = get_LLM_rep(model, args.model_type, task_items, prompt_syllogisms, prompt_transitive)
#score_syll = calculate_average_score(sub_list, layer_index_all, LLM_rep, fmri_all_sub, all_W_syll, all_W_tran, q_type='Syllogisms')
#score_tran = calculate_average_score(sub_list, layer_index_all, LLM_rep, fmri_all_sub, all_W_syll, all_W_tran, q_type='Transitive')
#logging.info('score syll: {:.3f}, score tran: {:.3f}'.format(score_syll, score_tran))
#print('score syll: {:.3f}, score tran: {:.3f}'.format(score_syll, score_tran))
cosine_syll = calculate_average_cosine(sub_list, layer_index_all, LLM_rep, fmri_all_sub, all_W_syll, all_W_tran, q_type='Syllogisms')
cosine_tran = calculate_average_cosine(sub_list, layer_index_all, LLM_rep, fmri_all_sub, all_W_syll, all_W_tran, q_type='Transitive')
logging.info('cosine syll: {:.3f}, cosine tran: {:.3f}'.format(cosine_syll, cosine_tran))
print('cosine syll: {:.3f}, cosine tran: {:.3f}'.format(cosine_syll, cosine_tran))



