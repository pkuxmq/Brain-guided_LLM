import os
import pandas as pd
import pickle
from LM import LM_nnsight
import numpy as np
import json
from hcp_utils import *
import argparse
import logging
import random


parser = argparse.ArgumentParser(description='Make intervention with general direction for HCP Relational Task')
parser.add_argument('-DATA', type=str, default='./data/relational_test_set.jsonl')
parser.add_argument('-RESULTS_DIR', type=str, default='./hcp_results/intervention_sumdir_hcp/')
parser.add_argument('-INTERVE_INFO_DIR', type=str, default='./hcp_results/intervention_results/')
parser.add_argument('-LLM_PREV_RESULTS_DIR', type=str, default='./hcp_results/activations_results/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/')
parser.add_argument('-bf16', action='store_true')

parser.add_argument('-analyze_type', type=str, default='md')
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
parser.add_argument('-select_nearest', action='store_true')

parser.add_argument('-manual_seed', type=int, default=0)

parser.add_argument('-device', type=str, default='cpu')

args = parser.parse_args()

np.random.seed(args.manual_seed)

if args.random:
    suffix = 'random'
elif args.random_fmri:
    suffix = 'randomfmri'
    if args.with_fmri_mean:
        suffix += 'withfmrimean'
else:
    suffix = args.analyze_type
if args.add_intercept:
    suffix += '_addintercept'
suffix += '_' + args.loss_type
suffix += '_lr' + str(args.lr) + '-iteration{:d}'.format(args.iteration)
suffix += '_projalpha{:.1f}'.format(args.proj_alpha)
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
if args.normalize_dir:
    suffix2 += 'normdir'
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
elif args.model_type in ['mistral', 'llama3']:
    model = LM_nnsight(model_path, device, bf16=args.bf16)
    layer_num = 32
elif args.model_type in ['qwen1-5b', 'qwen7b']:
    model = LM_nnsight(model_path, device, bf16=args.bf16)
    layer_num = 28
elif args.model_type == 'deepseekqwen1-5b':
    model = LM_nnsight(model_path, device, bf16=args.bf16)
    layer_num = 28
    max_new_tokens = 2000
elif args.model_type == 'phi4-mini':
    model = LM_nnsight(model_path, device, bf16=args.bf16)
    layer_num = 32
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

# Load Test Set
print(f"Loading test data from {args.DATA}")
test_dataset = []
with open(args.DATA, 'r') as f:
    for line in f:
        if line.strip():
            test_dataset.append(json.loads(line))
logging.info(f"Loaded {len(test_dataset)} test items.")

# Load LLM Previous Results (Activations) for Normalization (std calc)
print(f"Loading previous LLM activations from {LLM_PREV_RESULTS_DIR}")
path_activations = os.path.join(LLM_PREV_RESULTS_DIR, 'hcp_relational_activations.pkl')
LLM_rep_all = None

try:
    with open(path_activations, "rb") as f:
        # {stim_name: (Layers, Dim)}
        activations_dict = pickle.load(f)
    
    # Check shape of first item
    first_key = list(activations_dict.keys())[0]
    first_val = activations_dict[first_key]
    if isinstance(first_val, dict):
        attn = first_val.get('attention')
        hid = first_val.get('hidden')
        if attn is None or hid is None:
            raise ValueError("Activation dict missing 'attention' or 'hidden'.")
        first_shape = np.concatenate([attn, hid], axis=0).shape
    else:
        first_shape = first_val.shape
    logging.info(f"Activation shape for single item: {first_shape}")
    
    # Stack all: (N_samples, Layers(=2*layer), Dim)
    list_acts = []
    for k, v in activations_dict.items():
        if isinstance(v, dict):
            attn = v.get('attention')
            hid = v.get('hidden')
            if attn is None or hid is None:
                continue
            v = np.concatenate([attn, hid], axis=0)
        list_acts.append(v)
    
    LLM_rep_all = np.stack(list_acts, axis=0) # (N, Layers, Dim)
    logging.info(f"Constructed LLM_rep_all with shape: {LLM_rep_all.shape}")

except Exception as e:
    logging.warning(f"Failed to load LLM activations for STD calculation: {e}")
    logging.warning("Standard deviation normalization will be skipped or approximated if enabled.")

# Load Intervention Vectors
if not args.random:
    # Try loading question-aggregated direction file
    # hcp_make_intervention saves: aggregator_rep_diff_dir['Relational'][stim_name] = [vecs...]
    path_results = os.path.join(INTERVE_INFO_DIR, 'intervention_info_dir.pkl')
    if not os.path.exists(path_results):
        path_results = os.path.join(INTERVE_INFO_DIR, 'intervention_info.pkl')
    
    print(f"Loading intervention info from {path_results}")
    with open(path_results, 'rb') as f:
        intervention_info = pickle.load(f)
else:
    intervention_info = {}

# Search over Direction Scale
dir_scale = args.dir_scale_min
acc_all = []

target_types = ['Relational']

while dir_scale <= args.dir_scale_max + 1e-9:
    logging.info(f"Processing dir_scale: {dir_scale:.3f}")
    
    # 1. Compute Candidates
    candidates_per_layer = {l: [] for l in layer_index_all}
    q_type = 'Relational'
    
    if args.random:
         if LLM_rep_all is not None:
             dim = LLM_rep_all.shape[-1]
         else:
             # Fallback dim if model config is accessible via model.model.model
             try:
                dim = model.model.model.config.hidden_size
             except:
                # Fallback for some wrappers
                dim = model.model.config.hidden_size

         for layer in layer_index_all:
             rand_dir = np.random.randn(dim)
             rand_dir = rand_dir / np.linalg.norm(rand_dir)
             candidates_per_layer[layer].append(rand_dir)
    elif q_type in intervention_info:
        sub_dict = intervention_info[q_type]
        for stim_name, vec_list in sub_dict.items():
            if len(vec_list) == 0: continue
            
            target_list = []
            if args.select_nearest:
                # Find single best vector (smallest average norm across layers)
                best_intv = None
                min_norm = float('inf')
                 
                for intv_dict in vec_list:
                    curr_norm_sum = 0
                    for layer in layer_index_all:
                        curr_norm_sum += np.linalg.norm(intv_dict[layer])
                     
                    avg_norm = curr_norm_sum / len(layer_index_all)
                    if avg_norm < min_norm:
                        min_norm = avg_norm
                        best_intv = intv_dict
                 
                if best_intv is not None:
                    target_list = [best_intv]
            else:
                 target_list = vec_list
            
            # Add to candidates
            for intv_dict in target_list:
                for layer in layer_index_all:
                     if layer in intv_dict:
                         vec = intv_dict[layer].reshape(-1)
                         if args.normalize_dir:
                             n = np.linalg.norm(vec)
                             if n > 0: vec = vec/n
                         candidates_per_layer[layer].append(vec)

    # 2. Aggregate to form Intervention Dictionary
    direction_info_all = {'Relational': {}}
    for layer in layer_index_all:
        cand_list = candidates_per_layer[layer]
        if len(cand_list) > 0:
             stack_vec = np.stack(cand_list, axis=0) # (N, Dim)
             mean_dir = np.mean(stack_vec, axis=0)
             
             if args.normalize_dir and LLM_rep_all is not None:
                  # Normalize Mean Direction first
                  n = np.linalg.norm(mean_dir)
                  if n > 0: mean_dir_norm = mean_dir / n
                  else: mean_dir_norm = mean_dir
                  
                  # Scale by Population STD
                  states = LLM_rep_all[:, layer, :]
                  proj = states @ mean_dir_norm
                  std_val = np.std(proj)
                  final_dir = mean_dir_norm * std_val * dir_scale
             elif args.normalize_dir:
                  n = np.linalg.norm(mean_dir)
                  if n > 0: final_dir = (mean_dir / n) * dir_scale
                  else: final_dir = mean_dir * dir_scale
             else:
                  final_dir = mean_dir * dir_scale
             
             direction_info_all['Relational'][layer] = final_dir
        else:
             raise("No candidate directions found for layer {}".format(layer))

    # 3. Perform Intervention
    intervention_dict = direction_info_all['Relational']
    if not intervention_dict:
         logging.warning("Intervention Dictionary empty or invalid.")
    
    model_ans_all = []
    correct_all = []
    
    for i, item in enumerate(test_dataset):
        try:
             messages = construct_relational_prompt(item)
        except Exception as e:
             logging.error(f"Prompt construction failed for item {i}: {e}")
             continue

        if 'deepseek' in args.model_type.lower():
            with torch.no_grad():
                ans = model.intervention_multilayer(messages, intervention_dict, max_new_tokens=max_new_tokens, apply_all_tokens=True)
        else:
            with torch.no_grad():
                ans = model.intervention_multilayer(messages, intervention_dict, max_new_tokens=max_new_tokens)
        
        val = parse_relational_response(ans, model_type=args.model_type)
        
        label_raw = item.get('label', -1)
        label = -1
        if isinstance(label_raw, int): label = label_raw
        elif isinstance(label_raw, str):
            clean_l = label_raw.lower().strip()
            if clean_l in ['true', 'yes', '1']: label = 1
            elif clean_l in ['false', 'no', '0']: label = 0
            
        is_correct = (val == label)
        correct_all.append(is_correct)
        model_ans_all.append(val)
    
    acc = np.mean(correct_all) if len(correct_all) > 0 else 0.0
    
    logging.info('--------------------')
    logging.info('Direction scale {:.3f}'.format(dir_scale))
    logging.info('Overall acc is: {:.3f}'.format(acc))
    print('Direction scale {:.3f}, Overall acc: {:.3f}'.format(dir_scale, acc))

    acc_all.append(acc)

    dir_scale += args.dir_scale_itv

logging.info(acc_all)
if len(acc_all) > 0:
    acc_max = max(acc_all)
    logging.info('--------------------')
    logging.info('In sum, max acc is: {:.3f}'.format(acc_max))
    print('In sum, max acc is: {:.3f}'.format(acc_max))
else:
    print('No results.')
