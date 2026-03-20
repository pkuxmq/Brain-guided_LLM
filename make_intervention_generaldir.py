import os
import pandas as pd
import pickle
from LM import LM_nnsight
import numpy as np
from utils import *
import argparse
import logging
import random


parser = argparse.ArgumentParser(description='Make intervention with general direction')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/')
parser.add_argument('-DATA', type=str, default='./data/deductive_reasoning_data_test.json')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/intervention_sumdir_test/')
parser.add_argument('-INTERVE_INFO_DIR', type=str, default='./results/intervention_sep_results/')
parser.add_argument('-LLM_PREV_RESULTS_DIR', type=str, default='./results/activations_results/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/')
parser.add_argument('-bf16', action='store_true')
parser.add_argument('-prompt1', action='store_true')

parser.add_argument('-analyze_type', type=str, default='deductive_reasoning')
parser.add_argument('-loss_type', type=str, default='cosine')
parser.add_argument('-ridge_alpha', type=float, default=100.)
parser.add_argument('-use_ridgecv', action='store_true')

parser.add_argument('-lr', type=float, default=1e-1)
parser.add_argument('-iteration', type=int, default=200)
parser.add_argument('-intv_scale', type=float, default=1.)
parser.add_argument('-proj_alpha', type=float, default=10.)

parser.add_argument('-iter_interval', type=int, default=5)
parser.add_argument('-fit_consistent_questions', action='store_true')
parser.add_argument('-random', action='store_true')
parser.add_argument('-random_fmri', action='store_true')
parser.add_argument('-with_fmri_mean', action='store_true')
parser.add_argument('-all_layer', action='store_true')
parser.add_argument('-add_intercept', action='store_true')

parser.add_argument('-suffix', type=str, default='')

parser.add_argument('-dir_scale_min', type=float, default=0.1)
parser.add_argument('-dir_scale_max', type=float, default=1.)
parser.add_argument('-dir_scale_itv', type=float, default=0.1)
parser.add_argument('-normalize_dir', action='store_true')
parser.add_argument('-combine_dir', action='store_true')
parser.add_argument('-select_nearest', action='store_true')

parser.add_argument('-manual_seed', type=int, default=0)

parser.add_argument('-device', type=str, default='cpu')

args = parser.parse_args()
args.signal_type = 'beta'

np.random.seed(args.manual_seed)

DATA_DIR = args.DATA_DIR
if args.random:
    suffix = 'random'
elif args.random_fmri:
    suffix = 'randomfmri'
    if args.with_fmri_mean:
        suffix += 'withfmrimean'
else:
    suffix = args.analyze_type
suffix += '_' + args.loss_type
suffix += '_lr' + str(args.lr) + '-iteration{:d}'.format(args.iteration)
suffix += '_projalpha{:d}'.format(int(args.proj_alpha))
if args.intv_scale != 1.:
    suffix += '_intvscale{:.1f}'.format(args.intv_scale)
if args.fit_consistent_questions:
    suffix += '_ridgeforcon'
if args.all_layer:
    suffix += '_alllayer'
else:
    suffix += '_middlelayer'
if args.use_ridgecv:
    suffix += '_ridgecv'
    
INTERVE_INFO_DIR = os.path.join(args.INTERVE_INFO_DIR, args.model_type, suffix)
suffix2 = ''
if args.prompt1:
    suffix2 += 'prompt1'
if args.normalize_dir:
    suffix2 += 'normdir'
if args.combine_dir:
    suffix2 += 'combinedir'
if args.suffix != '':
    suffix2 += '_' + args.suffix
RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type, suffix2, suffix)
LLM_PREV_RESULTS_DIR = os.path.join(args.LLM_PREV_RESULTS_DIR, args.model_type)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
log_path = os.path.join(RESULTS_DIR, 'results.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')

device = args.device

max_new_tokens = 1
model_path = args.model_path
if args.model_type == 'llama2':
    model = LM_nnsight(model_path, device, bf16=args.bf16)
    layer_num = 32
    max_new_tokens = 2
elif args.model_type in ['mistral', 'llama3', 'phi4-mini']:
    model = LM_nnsight(model_path, device, bf16=args.bf16)
    layer_num = 32
elif args.model_type in ['qwen1-5b', 'qwen7b']:
    model = LM_nnsight(model_path, device, bf16=args.bf16)
    layer_num = 28
elif args.model_type == 'deepseekqwen1-5b':
    model = LM_nnsight(model_path, device, bf16=args.bf16)
    layer_num = 28
    max_new_tokens = 2000
elif args.model_type == 'gemma2_9b':
    model = LM_nnsight(model_path, device, bf16=args.bf16)
    layer_num = 42
else:
    raise("Error! Unsupported model type")

if args.all_layer:
    layer_index_all = list(range(0, layer_num))
else:
    layer_index_all = list(range(layer_num//4, layer_num*3//4))

logging.info('layer index all:')
logging.info(layer_index_all)

# load test set
dataset = load_from_json(args.DATA)

with open(os.path.join(DATA_DIR, 'task_items.pkl'), 'rb') as f:
    task_items = pickle.load(f)
task_run_list = sorted(task_items.keys())

# load prev LLM features
LLM_rep_all = load_LLM_features(LLM_PREV_RESULTS_DIR, task_run_list)

if not args.random:
    path_results = os.path.join(INTERVE_INFO_DIR, 'intervention_info_dir.pkl')
    if not os.path.exists(path_results):
        path_results = os.path.join(INTERVE_INFO_DIR, 'intervention_info.pkl')
    with open(path_results, 'rb') as f:
        intervention_info = pickle.load(f)

# model prompt
prompt_syllogisms = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe a series of relationships among monosyllabic pseudowords and adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."
prompt_transitive = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe relationships among imaginary characters with comparative adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."
prompt1 = "Three premises and one conclusion will be presented. Judge if the conclusion can be drawn from the premises logically. Answer 'True' or 'False'.\n"

# search over direction scale
dir_scale = args.dir_scale_min
syll_acc_all = []
tran_acc_all = []
while dir_scale <= args.dir_scale_max:
    # get a summed direction
    direction_info_all = {'Syllogisms': {}, 'Transitive': {}}
    if args.combine_dir:
        direction_info_all['All'] = {}
        for layer in layer_index_all:
            direction_info_all['All'][layer] = []
    for q_type in ['Syllogisms', 'Transitive']:
        for layer in layer_index_all:
            direction_info_all[q_type][layer] = []
        if args.random:
            for layer in layer_index_all:
                rand_dir = np.random.randn(*LLM_rep_all[0, layer].shape)
                rand_dir = rand_dir / np.linalg.norm(rand_dir)
                direction_info_all[q_type][layer].append(rand_dir)
                if args.combine_dir:
                    direction_info_all['All'][layer].append(rand_dir)
        else:
            for q_ind in intervention_info[q_type].keys():
                if not args.select_nearest:
                    for intv_info in intervention_info[q_type][q_ind]:
                        for layer in layer_index_all:
                            if not args.normalize_dir:
                                direction_info_all[q_type][layer].append(intv_info[layer])
                                if args.combine_dir:
                                    direction_info_all['All'][layer].append(intv_info[layer])
                            else:
                                direction = intv_info[layer] / np.linalg.norm(intv_info[layer])
                                direction_info_all[q_type][layer].append(direction)
                                if args.combine_dir:
                                    direction_info_all['All'][layer].append(direction)
                else:
                    # first identify the averagely nearest intervention point
                    min_avg_dis = None
                    for rep_diff in intervention_info[q_type][q_ind]:
                        avg_dis = 0
                        for layer in layer_index_all:
                            avg_dis += np.linalg.norm(rep_diff[layer])
                        avg_dis /= len(layer_index_all)
                        if min_avg_dis is None:
                            min_avg_dis = avg_dis
                            rep_diff_nearest = rep_diff
                        else:
                            if avg_dis < min_avg_dis:
                                min_avg_dis = avg_dis
                                rep_diff_nearest = rep_diff
                    # use the nearest point
                    for layer in layer_index_all:
                        if not args.normalize_dir:
                            direction_info_all[q_type][layer].append(rep_diff_nearest[layer])
                            if args.combine_dir:
                                direction_info_all['All'][layer].append(rep_diff_nearest[layer])
                        else:
                            direction = rep_diff_nearest[layer] / np.linalg.norm(rep_diff_nearest[layer])
                            direction_info_all[q_type][layer].append(direction)
                            if args.combine_dir:
                                direction_info_all['All'][layer].append(direction)
        for layer in layer_index_all:
            if len(direction_info_all[q_type][layer]) > 0:
                if not args.normalize_dir:
                    direction_info_all[q_type][layer] = np.mean(np.stack(direction_info_all[q_type][layer], axis=0), axis=0) * dir_scale
                else:
                    direction = np.mean(np.stack(direction_info_all[q_type][layer], axis=0), axis=0)
                    direction = direction / np.linalg.norm(direction)
                    # get std
                    proj_vals = LLM_rep_all[:, layer, :] @ direction
                    std = np.std(proj_vals)
                    direction_info_all[q_type][layer] = direction * std * dir_scale
    if args.combine_dir:
        for layer in layer_index_all:
            if not args.normalize_dir:
                direction_info_all['All'][layer] = np.mean(np.stack(direction_info_all['All'][layer], axis=0), axis=0) * dir_scale
            else:
                direction = np.mean(np.stack(direction_info_all['All'][layer], axis=0), axis=0)
                direction = direction / np.linalg.norm(direction)
                # get std
                proj_vals = LLM_rep_all[:, layer, :] @ direction
                std = np.std(proj_vals)
                direction_info_all['All'][layer] = direction * std * dir_scale
    
    # perform intervention
    model_ans_all = []
    label_all = []
    correct_all = []
    question_types = ['syllogisms', 'transitive']
    for q_type in question_types:
        for data in dataset[q_type]:
            prompt = prompt_syllogisms
            if q_type == 'syllogisms':
                intervention_dict = direction_info_all['Syllogisms']
            else:
                intervention_dict = direction_info_all['Transitive']
            if args.combine_dir:
                intervention_dict = direction_info_all['All']
            # check if there is intervention information
            if all(len(v) == 0 for v in intervention_dict.values()):
                correct_all.append(-1)
                continue
            if args.prompt1:
                messages = [
                        {"role": "user", "content": prompt1 + "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (data['premise1'].strip(), data['premise2'].strip(), data['premise3'].strip(), data['conclusion'].strip())},
                ]
            else:
                messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                        {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (data['premise1'].strip(), data['premise2'].strip(), data['premise3'].strip(), data['conclusion'].strip())},
                ]

            if args.model_type in ['deepseekqwen1-5b']:
                with torch.no_grad():
                    ans = model.intervention_multilayer(messages, intervention_dict, max_new_tokens=max_new_tokens, apply_all_tokens=True)
                idx = ans.rfind('</think>')
                if idx != -1:
                    ans = ans[idx+len('</think>'):].strip('\n').strip()
                    if args.model_type in ['deepseekqwen1-5b']:
                        idx_t = ans.rfind('True')
                        if idx_t == -1:
                            idx_t = ans.rfind('true')
                        idx_f = ans.rfind('False')
                        if idx_f == -1:
                            idx_f = ans.rfind('false')
                        if idx_t != -1 and idx_f == -1:
                            ans = 'True'
                        elif idx_f != -1 and idx_t == -1:
                            ans = 'False'
                        elif idx_f == -1 and idx_t == -1:
                            idx_c = ans.rfind('can be drawn')
                            if idx_c == -1:
                                idx_c = ans.rfind('can be logically drawn')
                            if idx_c != -1:
                                ans = 'True'
                            else:
                                if 'can not' in ans or 'cannot' in ans or 'does not' in ans:
                                    ans = 'False'
                else:
                    print('Not end thinking')
            else:
                with torch.no_grad():
                    ans = model.intervention_multilayer(messages, intervention_dict, max_new_tokens=max_new_tokens)
                    ans = ans.strip().strip('</s>') # for llama2
        
            if ans in ['True', 'true']:
                model_ans = 1
            elif ans in ['False', 'false']:
                model_ans = 0
            else:
                model_ans = None
            model_ans_all.append(model_ans)
        
            trial_type = data['trial_type']
            if 'true' in trial_type:
                label = 1
            else: 
                label = 0
            label_all.append(label)
            correct_all.append(model_ans==label)
    
    logging.info('--------------------')
    logging.info('Direction scale {:.3f}'.format(dir_scale))
    logging.info('Overall acc is: {:.3f}, syllogisms acc is: {:.3f}, transitive acc is: {:.3f}'.format(average_list(correct_all), average_list(correct_all[:len(dataset['syllogisms'])]), average_list(correct_all[len(dataset['syllogisms']):])))
    print('Direction scale {:.3f}'.format(dir_scale))
    print('Overall acc is: {:.3f}, syllogisms acc is: {:.3f}, transitive acc is: {:.3f}'.format(average_list(correct_all), average_list(correct_all[:len(dataset['syllogisms'])]), average_list(correct_all[len(dataset['syllogisms']):])))

    syll_acc_all.append(average_list(correct_all[:len(dataset['syllogisms'])]))
    tran_acc_all.append(average_list(correct_all[len(dataset['syllogisms']):]))
        
    categories = [
        "2_true_affirm", "2_false_affirm",
        "3_true_affirm", "3_false_affirm",
        "2_true_negate", "2_false_negate",
        "3_true_negate", "3_false_negate"
    ]
    trial_num_per_type = 100
    logging.info('Syllogisms:')
    for i in range(len(categories)):
        idx_start = i * trial_num_per_type
        idx_end = (i + 1) * trial_num_per_type
        logging.info(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
    logging.info('Transitive:')
    for i in range(len(categories)):
        idx_start = (i + len(categories)) * trial_num_per_type
        idx_end = (i + 1 + len(categories)) * trial_num_per_type
        logging.info(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))

    dir_scale += args.dir_scale_itv

logging.info(syll_acc_all)
logging.info(tran_acc_all)
syll_acc_max = max(syll_acc_all)
tran_acc_max = max(tran_acc_all)
logging.info('--------------------')
logging.info('In sum, max syllogisms acc is: {:.3f}, max transitive acc is: {:.3f}'.format(syll_acc_max, tran_acc_max))
print('In sum, max syllogisms acc is: {:.3f}, max transitive acc is: {:.3f}'.format(syll_acc_max, tran_acc_max))
