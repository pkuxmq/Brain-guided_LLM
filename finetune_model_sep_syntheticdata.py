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

parser = argparse.ArgumentParser(description='Finetune model')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/finetune_results_syntheticdata/')
parser.add_argument('-LLM_PREV_RESULTS_DIR', type=str, default='./results/activations_results/')
parser.add_argument('-FMRI_RESULTS_DIR', type=str, default='./fmri_data/preprocessed_data_glmsinglesep_newdrroi_topksep/top-10%/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/')

# direction information related
# intervention results related
parser.add_argument('-use_intervention_info', action='store_true') # NARF; for NARF+Label, do not use intervention information
parser.add_argument('-INTERVENTION_RESULTS_DIR', type=str, default='./results/intervention_sep_results/')
# intervention hyperparameters for loading
parser.add_argument('-interv_lr', type=float, default=1e-1)
parser.add_argument('-iteration', type=int, default=200)
parser.add_argument('-iter_interval', type=int, default=5)
parser.add_argument('-proj_alpha', type=float, default=10.)
parser.add_argument('-fit_consistent_questions', action='store_true')
# shared hyperparameter
parser.add_argument('-analyze_type', type=str, default='deductive_reasoning')
parser.add_argument('-loss_type', type=str, default='cosine')
parser.add_argument('-ridge_alpha', type=float, default=100.)
parser.add_argument('-use_ridgecv', action='store_true')
parser.add_argument('-all_layer', action='store_true') # we use middle layers by default

parser.add_argument('-select_nearest', action='store_true')
parser.add_argument('-scale_dir', type=float, default=1.)

parser.add_argument('-use_random_fmri', action='store_true')
parser.add_argument('-use_label_as_fmri', action='store_true')
parser.add_argument('-use_fmri_for_correct', action='store_true')

parser.add_argument('-add_intercept', action='store_true')

# optimization related
parser.add_argument('-lora', action='store_true')
parser.add_argument('-bf16', action='store_true')
# set forward partial for acceleration
parser.add_argument('-forward_partial', action='store_true')

# combination with label supervision
parser.add_argument('-use_label', action='store_true')
parser.add_argument('-label_start_epoch', type=int, default=-1)
parser.add_argument('-label_end_epoch', type=int, default=-1)
parser.add_argument('-balance_label_class', action='store_true')
parser.add_argument('-label_loss_weight', type=float, default=1.)
parser.add_argument('-label_epoch_freq', type=int, default=1)

# training settings
parser.add_argument('-max_epoch', type=int, default=100)
parser.add_argument('-lr', type=float, default=1e-6)
parser.add_argument('-weight_decay', type=float, default=0.)
parser.add_argument('-train_weight', type=float, default=1.)
parser.add_argument('-train_weight_tran', type=float, default=None)
parser.add_argument('-reg_weight', type=float, default=0.)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-test_batch_size', type=int, default=1)
parser.add_argument('-grad_accumulate_step', type=int, default=1)
parser.add_argument('-scheduler', action='store_true')

parser.add_argument('-val_epoch', type=int, default=10)
parser.add_argument('-val_start_epoch', type=int, default=0)
parser.add_argument('-early_stop_tolerance', type=int, default=10) #4

parser.add_argument('-resume_model_path', type=str, default=None)
parser.add_argument('-resume_optimizer_path', type=str, default=None)
parser.add_argument('-resume_checkpoint_path', type=str, default=None)

parser.add_argument('-only_syll', action='store_true')
parser.add_argument('-only_tran', action='store_true')

parser.add_argument('-val_all_order', action='store_true')

parser.add_argument('-filter_sub_fmri', action='store_true')
parser.add_argument('-filter_sub_fmri_model_specific', action='store_true')

parser.add_argument('-high_sub_num', type=int, default=None)
parser.add_argument('-low_sub_num', type=int, default=None)

# suffix for log directory
parser.add_argument('-suffix', type=str, default='')

# validation and test datasets
parser.add_argument('-VAL_DATA', type=str, default='./data/deductive_reasoning_data_val_new.json')
parser.add_argument('-TEST_DATA', type=str, default='./data/deductive_reasoning_data_test.json')
parser.add_argument('-synthetic_data_file', type=str, default=None)

# device
parser.add_argument('-device', type=str, default='cpu')
parser.add_argument('-w_device', type=str, default=None)

parser.add_argument('-delete_checkpoint', action='store_true')
parser.add_argument('-test_origin', action='store_true')
parser.add_argument('-log_stdout', action='store_true', help='Also print logs to stdout')
parser.add_argument('-seed', type=int, default=0)

args = parser.parse_args()

_seed_ = args.seed
random.seed(_seed_)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(_seed_)

DATA_DIR = args.DATA_DIR
if args.use_intervention_info:
    # get intervention info directory
    suffix = args.analyze_type
    suffix += '_' + args.loss_type
    suffix += '_lr' + str(args.interv_lr) + '-iteration{:d}'.format(args.iteration)
    suffix += '_projalpha{:d}'.format(int(args.proj_alpha))
    if args.fit_consistent_questions:
        suffix += '_ridgeforcon'
    if args.all_layer:
        suffix += '_alllayer'
    else:
        suffix += '_middlelayer'
    if args.use_ridgecv:
        suffix += '_ridgecv'
    INTERVENTION_RESULTS_DIR = os.path.join(args.INTERVENTION_RESULTS_DIR, args.model_type, suffix)
    path_results = os.path.join(INTERVENTION_RESULTS_DIR, f'intervention_info.pkl')
    with open(path_results, 'rb') as f:
        intervention_info = pickle.load(f)

    if args.select_nearest:
        suffix += '_nearest'

    suffix = 'useintvinfo_' + suffix
    if args.scale_dir != 1.:
        suffix += f'_scaledir{args.scale_dir}'
else:
    suffix = 'nointvinfo'
    if args.loss_type != 'cosine':
        suffix += args.loss_type
    if args.use_fmri_for_correct:
        suffix += '-allfmri'
    if args.filter_sub_fmri:
        suffix += '-filtersub'
    if args.filter_sub_fmri_model_specific:
        suffix += '-msfiltersub'
    if args.add_intercept:
        suffix += '_addintercept'

if args.use_random_fmri:
    suffix += '_randfmri'
if args.use_label_as_fmri:
    suffix += '_labelasfmri'

if args.only_syll:
    suffix = 'onlysyll_' + suffix
if args.only_tran:
    suffix = 'onlytran_' + suffix

if args.lora:
    suffix += '_lora'
if args.bf16:
    suffix += '_bf16'
if args.use_label:
    if args.label_start_epoch < 0:
        args.label_start_epoch = 0
    if args.label_end_epoch < 0:
        args.label_end_epoch = args.max_epoch
    suffix += f'_uselabelfreq{args.label_epoch_freq}-{args.label_start_epoch}-{args.label_end_epoch}'
    if args.label_loss_weight != 1.:
        suffix += f'-labelweight{args.label_loss_weight}'
    if args.balance_label_class:
        suffix += 'balance-class'
suffix += f'_lr{args.lr}'
if args.weight_decay > 0:
    suffix += f'-wd'
suffix += f'_lossw-{int(args.reg_weight)}'
if args.train_weight != 1.:
    suffix += f'-train{args.train_weight}'
if args.train_weight_tran is None:
    args.train_weight_tran = args.train_weight
else:
    suffix += f'-traintran{args.train_weight_tran}'
if args.grad_accumulate_step > 1:
    suffix += f'_gradstep-{args.grad_accumulate_step}'
if args.scheduler:
    suffix += f'_scheduler'
if args.val_all_order:
    suffix += '_valallorder'

if args.suffix != '':
    suffix += '_' + args.suffix

RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type, suffix)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

log_path = os.path.join(RESULTS_DIR, 'train.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if args.log_stdout:
    logging.getLogger().addHandler(logging.StreamHandler())
logging.info(args)
device = args.device

max_new_tokens = 1
parse_model_type = 'default'
model_path = args.model_path
if args.model_type == 'llama2':
    parse_model_type = 'llama2'
    layer_num = 32
    max_new_tokens = 2
elif args.model_type == 'llama3':
    layer_num = 32
elif args.model_type == 'mistral':
    layer_num = 32
elif args.model_type == 'qwen1-5b':
    layer_num = 28
elif args.model_type == 'qwen7b':
    layer_num = 28
elif args.model_type == 'qwen72b':
    layer_num = 80
elif args.model_type == 'llama3-3_70b':
    layer_num = 80
elif args.model_type == 'qwen3_4b':
    parse_model_type = 'qwen3'
    layer_num = 36
elif args.model_type == 'phi4-mini':
    layer_num = 32
elif args.model_type == 'gemma2_9b':
    layer_num = 42
else:
    raise("Error! Unsupported model type")

if args.all_layer:
    layer_index_all = list(range(0, layer_num))
else:
    layer_index_all = list(range(layer_num//4, layer_num*3//4))

logging.info('layer index for training:')
logging.info(layer_index_all)

if args.resume_checkpoint_path is not None:
    args.resume_model_path = os.path.join(args.resume_checkpoint_path, 'model-max.pth')
    args.resume_optimizer_path = os.path.join(args.resume_checkpoint_path, 'optimizer-max.pth')

# define model and optimizer
model = ModelwithAttentionSupervision(model_path, layer_index_all, device, False, args.lora, args.bf16, args.model_type)
optimizer_parameters = []
for n, p in model.model.named_parameters():
    if p.requires_grad:
        optimizer_parameters.append(p)
optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr, weight_decay=args.weight_decay)

# load model and optimizer
if args.resume_model_path is not None:
    model_state_dict = torch.load(args.resume_model_path, map_location='cpu')
    model.model.load_state_dict(model_state_dict)
if args.resume_optimizer_path is not None:
    optimizer.load_state_dict(torch.load(args.resume_optimizer_path, map_location=torch.device('cpu')))
    for g in optimizer.param_groups:
        g['lr'] = args.lr

# load representations and fmri
LLM_PREV_RESULTS_DIR = os.path.join(args.LLM_PREV_RESULTS_DIR, args.model_type)
FMRI_RESULTS_DIR = os.path.join(args.FMRI_RESULTS_DIR, 'all_extracted_beta_' + args.analyze_type)

filter_index = ['sub-1010', 'sub-1016', 'sub-1026', 'sub-1031', 'sub-1032', 'sub-1035', 'sub-1021']
if args.filter_sub_fmri:
    filter_index += ['sub-1004', 'sub-1014', 'sub-1034']
if args.filter_sub_fmri_model_specific:
    if args.model_type == 'qwen1-5b':
        filter_index += ['sub-1004', 'sub-1014', 'sub-1019']
    elif args.model_type == 'qwen7b':
        filter_index += ['sub-1022', 'sub-1034']
    elif args.model_type == 'mistral':
        filter_index += ['sub-1001', 'sub-1022', 'sub-1027']
    elif args.model_type == 'llama2':
        filter_index += ['sub-1008', 'sub-1014', 'sub-1034']
    elif args.model_type == 'llama3':
        filter_index += ['sub-1014', 'sub-1019', 'sub-1034']
    elif args.model_type == 'phi4-mini':
        filter_index += ['sub-1004', 'sub-1014', 'sub-1017']
    elif args.model_type == 'gemma2_9b':
        filter_index += ['sub-1004', 'sub-1014', 'sub-1017']

acc_sorted_sub_index = ['sub-1030', 'sub-1027', 'sub-1019', 'sub-1001', 'sub-1017', 'sub-1004', 'sub-1034', 'sub-1014', 'sub-1022', 'sub-1008']

task_items, data_dict, sub_list, model_ans_all, label_all, correct_all = load_and_filter_behavior_results(DATA_DIR, LLM_PREV_RESULTS_DIR, filter_index)

if args.high_sub_num is not None:
    sub_list = acc_sorted_sub_index[:args.high_sub_num]
if args.low_sub_num is not None:
    sub_list = acc_sorted_sub_index[-args.low_sub_num:]

task_run_list = sorted(task_items.keys())
# get behavior results
model_acc, sub_task_acc = get_model_human_behavior(correct_all, data_dict, sub_list)
# load prev LLM features
LLM_rep_all = load_LLM_features(LLM_PREV_RESULTS_DIR, task_run_list)
# load all fmri data
fmri_all_sub, _ = get_all_fmri_data_latest(sub_list, FMRI_RESULTS_DIR)
if args.use_random_fmri:
    fmri_all_sub = np.random.randn(*fmri_all_sub.shape)
if args.use_label_as_fmri:
    #fmri_all_sub = np.zeros((10, 70, 2))
    fmri_all_sub = np.zeros((fmri_all_sub.shape[0], fmri_all_sub.shape[1], 2))
    for i in range(70):
        if i < 18:
            task_run = list(filter(lambda item: 'Syllogisms' in item and '01' in item, task_run_list))[0]
            fmri_all_sub[:, i, label_all[task_run][i]] = 1
        elif i < 36:
            task_run = list(filter(lambda item: 'Syllogisms' in item and '02' in item, task_run_list))[0]
            fmri_all_sub[:, i, label_all[task_run][i-18]] = 1
        elif i < 52:
            task_run = list(filter(lambda item: 'Transitive' in item and '01' in item, task_run_list))[0]
            fmri_all_sub[:, i, label_all[task_run][i-36]] = 1
        else:
            task_run = list(filter(lambda item: 'Transitive' in item and '02' in item, task_run_list))[0]
            fmri_all_sub[:, i, label_all[task_run][i-52]] = 1


# model prompt
prompt_syllogisms = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe a series of relationships among monosyllabic pseudowords and adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."
prompt_transitive = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe relationships among imaginary characters with comparative adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."

# get representation direction
signal_dict = []
index_for_training = []
index_for_regularization = []
trainreg_dir_info = {}
if args.use_intervention_info:
    for ind in range(LLM_rep_all.shape[0]):
        if ind < 36:
            q_type = 'Syllogisms'
            q_ind = ind
        else:
            q_type = 'Transitive'
            q_ind = ind - 36
        if q_ind in intervention_info[q_type].keys():
            # training question
            index_for_training.append(ind)
            direction_info_all = {}
            if not args.select_nearest:
                for layer in layer_index_all:
                    direction_info_all[layer] = []
                    for rep_diff in intervention_info[q_type][q_ind]:
                        rep_sup = rep_diff[layer] * args.scale_dir + LLM_rep_all[ind][layer]
                        direction_info_all[layer].append(rep_sup)
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
                    direction_info_all[layer] = rep_diff_nearest[layer] * args.scale_dir + LLM_rep_all[ind][layer]
            trainreg_dir_info[ind] = direction_info_all
        elif model_acc[ind]:
            # regularization question
            index_for_regularization.append(ind)
            direction_info_all = {}
            for layer in layer_index_all:
                direction_info_all[layer] = LLM_rep_all[ind][layer]
            trainreg_dir_info[ind] = direction_info_all
else:
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
            w_device = args.w_device if args.w_device is not None else args.device
            if args.bf16:
                W_syll = torch.from_numpy(W_syll).to(torch.bfloat16).to(w_device)
                W_tran = torch.from_numpy(W_tran).to(torch.bfloat16).to(w_device)
                if args.add_intercept:
                    b_syll = torch.from_numpy(b_syll).to(torch.bfloat16).to(w_device)
                    b_tran = torch.from_numpy(b_tran).to(torch.bfloat16).to(w_device)
            else:
                W_syll = torch.from_numpy(W_syll).float().to(w_device)
                W_tran = torch.from_numpy(W_tran).float().to(w_device)
                if args.add_intercept:
                    b_syll = torch.from_numpy(b_syll).float().to(w_device)
                    b_tran = torch.from_numpy(b_tran).float().to(w_device)
            all_W_syll[sub][layer] = W_syll
            all_W_tran[sub][layer] = W_tran
            if args.add_intercept:
                all_b_syll[sub][layer] = b_syll
                all_b_tran[sub][layer] = b_tran

    fmri_all_sub = torch.from_numpy(fmri_all_sub).float().to(args.device)
    for ind in range(LLM_rep_all.shape[0]):
        direction_info_all = {}
        if not args.use_fmri_for_correct and model_acc[ind]:
            # regularization
            index_for_regularization.append(ind)
            for layer in layer_index_all:
                direction_info_all[layer] = LLM_rep_all[ind][layer]
            trainreg_dir_info[ind] = direction_info_all
        else:
            ind_train = False
            for sub_ind, sub in enumerate(sub_list):
                if (not model_acc[ind] or args.use_fmri_for_correct) and sub_task_acc[sub]['overall'][ind]:
                    ind_train = True
                    for layer in layer_index_all:
                        if ind < 36:
                            W = all_W_syll[sub][layer]
                            if args.add_intercept:
                                b = all_b_syll[sub][layer]
                        else:
                            W = all_W_tran[sub][layer]
                            if args.add_intercept:
                                b = all_b_tran[sub][layer]
                        if args.add_intercept:
                            sup_dict = {'W': W, 'fmri_state': fmri_all_sub[sub_ind, ind], 'loss_type': args.loss_type, 'b': b}
                        else:
                            sup_dict = {'W': W, 'fmri_state': fmri_all_sub[sub_ind, ind], 'loss_type': args.loss_type}
                        if layer in direction_info_all.keys():
                            direction_info_all[layer].append(sup_dict)
                        else:
                            direction_info_all[layer] = [sup_dict]
            if ind_train:
                index_for_training.append(ind)
                trainreg_dir_info[ind] = direction_info_all


# get train data
num_for_training = len(index_for_training)
num_for_regularization = len(index_for_regularization)

question_index = 0
num_true_all = 0
num_false_all = 0
for task_run in sorted(task_items.keys()):
    df = task_items[task_run]
    train_q = True
    if 'Transitive' in task_run:
        prompt = prompt_transitive
        if args.only_syll:
            train_q = False
    else:
        prompt = prompt_syllogisms
        if args.only_tran:
            train_q = False
        
    for i in range(len(df)):
        messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (df['premise1'][i].strip(), df['premise2'][i].strip(), df['premise3'][i].strip(), df['conclusion'][i].strip())},
        ]

        if 'Transitive' in task_run and '01' in task_run and (i == 7 or i == 13):
            continue
        else:
            # get label
            if 'true' in df['trial_type'][i]:
                label = 'True'
                num_true_all += 1
            else:
                label = 'False'
                num_false_all += 1
            if question_index in index_for_training and train_q:
                signal_dict.append({'x': messages, 'y': trainreg_dir_info[question_index], 'label': label, 'type': 'train', 'q_index': question_index})
            elif question_index in index_for_regularization and train_q:
                signal_dict.append({'x': messages, 'y': trainreg_dir_info[question_index], 'label': label, 'type': 'reg', 'q_index': question_index})
    
            question_index += 1

if args.synthetic_data_file is not None:
    synthetic_data = load_from_json(args.synthetic_data_file)
    q_type_all = ['syllogisms', 'transitive']
    for q_type in q_type_all:
        if q_type == 'syllogisms':
            if args.only_tran:
                continue
            prompt = prompt_syllogisms
        if q_type == 'transitive':
            if args.only_syll:
                continue
            prompt = prompt_transitive
        
        for data in synthetic_data[q_type]:
            messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                    {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (data['premise1'].strip(), data['premise2'].strip(), data['premise3'].strip(), data['conclusion'].strip())},
            ]
            if 'true' in data['trial_type']:
                label = 'True'
                num_true_all += 1
            else:
                label = 'False'
                num_false_all += 1
            signal_dict.append({'x': messages, 'y': None, 'label': label, 'type': 'synthetic', 'q_index': -1})

batch_size = args.batch_size
# now only support batch size 1
assert batch_size == 1
grad_accumulate_step = args.grad_accumulate_step
if args.scheduler:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

# validation dataset
val_dataset = load_from_json(args.VAL_DATA)
val_acc_max = 0.
val_max_epoch = 0
# for early stop
prev_val_acc = 0.
decrease_cnt = 0
# test dataset
test_dataset = load_from_json(args.TEST_DATA)

if args.test_origin:
    model.model.eval()
    if args.test_batch_size == 1:
        acc, syl_acc, tra_acc = test_model(model, val_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, parse_model_type, args.only_syll, args.only_tran, args.val_all_order)
    else:
        # may slightly different for Mistral
        acc, syl_acc, tra_acc = test_model_batch(model, val_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, args.test_batch_size, parse_model_type, args.only_syll, args.only_tran, args.val_all_order)
    logging.info('--------------------')
    logging.info('Validation overall acc is: {:.3f}, syllogisms acc is: {:.3f}, transitive acc is: {:.3f}'.format(acc, syl_acc, tra_acc))
    logging.info('--------------------')
    if args.test_batch_size == 1:
        acc, syl_acc, tra_acc = test_model(model, test_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, parse_model_type, False, False, True)
    else:
        acc, syl_acc, tra_acc = test_model_batch(model, test_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, args.test_batch_size, parse_model_type, False, False, True)
    logging.info('--------------------')
    logging.info('Test overall acc is: {:.3f}, syllogisms acc is: {:.3f}, transitive acc is: {:.3f}'.format(acc, syl_acc, tra_acc))
    logging.info('--------------------')


# training
current_iteration = 0
optimizer.zero_grad()
for e in range(args.max_epoch):
    if (e + 1) % 10 == 0:
        print('epoch ' + str(e))
    logging.info('epoch ' + str(e))
    model.model.train()

    iter_indices = list(range(len(signal_dict)))
    random.shuffle(iter_indices)
    iter_num = (len(signal_dict)-1) // batch_size + 1

    total_loss = 0.
    train_loss = 0.
    reg_loss = 0.
    if args.use_label and e >= args.label_start_epoch and e < args.label_end_epoch and (e + 1) % args.label_epoch_freq == 0:
        label_loss = 0.
        all_q_index = []
        num_true = 0
        num_false = 0
        max_num_each = min(num_true_all, num_false_all)

    for idx in iter_indices:
        # get inputs
        sample = signal_dict[idx]
        x = sample['x']
        if args.model_type == 'llama2':
            text = model.tokenizer.apply_chat_template(x, tokenize=False)
            text += ' '
            encodeds = model.tokenizer([text], return_tensors='pt').to(device)
            input_ids = encodeds.input_ids
        elif 'qwen3' in args.model_type:
            encodeds = model.tokenizer.apply_chat_template(x, return_tensors='pt', add_generation_prompt=True, enable_thinking=False)
            input_ids = encodeds.to(device)
        else:
            encodeds = model.tokenizer.apply_chat_template(x, return_tensors='pt', add_generation_prompt=True)
            input_ids = encodeds.to(device)
        attention_mask = None
        # get sup info
        y = sample['y']
        # get label info
        label = sample['label']
        label = model.tokenizer(label, return_tensors='pt').to(device)
        label = label['input_ids'].flatten()
        # avoid problem from mistral tokenizer
        label = label[-1:]

        sample_type = sample['type']
        if sample_type == 'train':
            loss_weight = args.train_weight
            if args.train_weight_tran != args.train_weight:
                if sample['q_index'] >= 36:
                    loss_weight = args.train_weight_tran
        elif sample_type == 'reg':
            loss_weight = args.reg_weight
        else:
            loss_weight = 0.

        if loss_weight != 0 and y is not None:
            if args.bf16:
                with autocast(dtype=torch.bfloat16):
                    if args.forward_partial:
                        loss = model.forward_partial(input_ids, attention_mask, y) * loss_weight / grad_accumulate_step
                    else:
                        loss = model(input_ids, attention_mask, y) * loss_weight / grad_accumulate_step
            else:
                if args.forward_partial:
                    loss = model.forward_partial(input_ids, attention_mask, y) * loss_weight / grad_accumulate_step
                else:
                    loss = model(input_ids, attention_mask, y) * loss_weight / grad_accumulate_step
            if sample_type == 'train':
                train_loss += loss.item()
            else:
                reg_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

        if args.use_label and e >= args.label_start_epoch and e < args.label_end_epoch and (e + 1) % args.label_epoch_freq == 0:
            if not args.balance_label_class:
                if args.bf16:
                    with autocast(dtype=torch.bfloat16):
                        loss_label = model(input_ids, attention_mask, y, label) * args.label_loss_weight / grad_accumulate_step
                else:
                    loss_label = model(input_ids, attention_mask, y, label) * args.label_loss_weight / grad_accumulate_step
                label_loss += loss_label.item()
                    
                loss_label.backward()
            else:
                q_index = sample['q_index']
                q_label = sample['label']
                if q_label == 'True':
                    num_true += 1
                else:
                    num_false += 1
                if (q_label == 'True' and num_true <= max_num_each) or (q_label == 'False' and num_false <= max_num_each):
                    if args.bf16:
                        with autocast(dtype=torch.bfloat16):
                            loss_label = model(input_ids, attention_mask, y, label) * args.label_loss_weight / grad_accumulate_step
                    else:
                        loss_label = model(input_ids, attention_mask, y, label) * args.label_loss_weight / grad_accumulate_step
                    label_loss += loss_label.item()

                    loss_label.backward()
                    all_q_index.append(q_index)

        current_iteration += 1
        if current_iteration % grad_accumulate_step == 0:
            optimizer.step()
            optimizer.zero_grad()

    total_loss /= len(signal_dict)
    train_loss /= num_for_training
    if num_for_regularization > 0:
        reg_loss /= num_for_regularization
    total_loss *= grad_accumulate_step
    train_loss *= grad_accumulate_step
    reg_loss *= grad_accumulate_step
    if args.use_label and e >= args.label_start_epoch and e < args.label_end_epoch and (e + 1) % args.label_epoch_freq == 0:
        if not args.balance_label_class:
            label_loss /= len(signal_dict)
        else:
            label_loss /= len(all_q_index)
        label_loss *= grad_accumulate_step
        logging.info('loss {:.6f}, train loss {:.6f}, reg loss {:.6f}, label loss {:.6f}'.format(total_loss, train_loss, reg_loss, label_loss))
    else:
        logging.info('loss {:.6f}, train loss {:.6f}, reg loss {:.6f}'.format(total_loss, train_loss, reg_loss))
    if args.scheduler:
        lr_scheduler.step()

    # validation
    if (e + 1) >= args.val_start_epoch and (e + 1) % args.val_epoch == 0:
        model.model.eval()
        if args.test_batch_size == 1:
            acc, syl_acc, tra_acc = test_model(model, val_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, parse_model_type, args.only_syll, args.only_tran, args.val_all_order)
        else:
            acc, syl_acc, tra_acc = test_model_batch(model, val_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, args.test_batch_size, parse_model_type, args.only_syll, args.only_tran, args.val_all_order)
        logging.info('--------------------')
        logging.info('Validation overall acc is: {:.3f}, syllogisms acc is: {:.3f}, transitive acc is: {:.3f}'.format(acc, syl_acc, tra_acc))
        logging.info('--------------------')
        if acc > val_acc_max:
            val_acc_max = acc
            val_max_epoch = e + 1
            save_model_path = os.path.join(RESULTS_DIR, 'model-max.pth')
            torch.save(model.model.state_dict(), save_model_path)
            save_optimizer_path = os.path.join(RESULTS_DIR, 'optimizer-max.pth')
            torch.save(optimizer.state_dict(), save_optimizer_path)

        # early stop
        if acc < prev_val_acc:
            decrease_cnt += 1
        elif acc > prev_val_acc:
            decrease_cnt = 0
        if decrease_cnt == args.early_stop_tolerance:
            logging.info('Early stop at epoch ' + str(e + 1))
            break
        prev_val_acc = acc

# use the best model on validation
logging.info(f'loading model from the epoch {val_max_epoch}')
save_model_path = os.path.join(RESULTS_DIR, 'model-max.pth')
model_state_dict = torch.load(save_model_path, map_location='cpu')
model.model.load_state_dict(model_state_dict)

if args.delete_checkpoint:
    # delete checkpoint
    os.remove(os.path.join(RESULTS_DIR, 'model-max.pth'))
    os.remove(os.path.join(RESULTS_DIR, 'optimizer-max.pth'))

# save model
save_model_path = os.path.join(RESULTS_DIR, 'model')
# merge lora
if args.lora:
    model.merge_lora()
model.model.save_pretrained(save_model_path)
model.tokenizer.save_pretrained(save_model_path)

# test finetuned model on fMRI dataset
model.model.eval()
finetune_model_ans_all = {}
finetune_correct_all = {}
for task_run in sorted(task_items.keys()):
    df = task_items[task_run]
    finetune_model_ans_all[task_run] = []
    finetune_correct_all[task_run] = []
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
        with torch.no_grad():
            ans = model.generate_ans(messages, max_new_tokens=max_new_tokens, parse_model_type=parse_model_type)
        if ans in ['True', 'true', 'True.']:
            model_ans = 1
        elif ans in ['False', 'false', 'False.']:
            model_ans = 0
        else:
            model_ans = None
            logging.info('----------')
            logging.info(task_run + ' problem ' + str(i))
            logging.info('No answer. The output is ' + ans)
            logging.info('----------')
        finetune_model_ans_all[task_run].append(model_ans)
    
        trial_type = df['trial_type'][i]
        if 'true' in trial_type:
            label = 1
        else:
            label = 0
        finetune_correct_all[task_run].append(model_ans==label)

path_results = os.path.join(RESULTS_DIR, f'behaviour_results.pkl')
with open(path_results, "wb") as f:
    pickle.dump([finetune_model_ans_all, label_all, finetune_correct_all], f)
    
# analyze
# filter contradictory questions
for task_run in sorted(finetune_model_ans_all.keys()):
    if 'Transitive' in task_run and '01' in task_run:
        del finetune_model_ans_all[task_run][13]
        del finetune_model_ans_all[task_run][7]
        del finetune_correct_all[task_run][13]
        del finetune_correct_all[task_run][7]

# accuracy
finetune_model_syllogisms_acc = []
finetune_model_transitive_acc = []
for task_run in sorted(finetune_correct_all.keys()):
    if 'Transitive' in task_run:
        finetune_model_transitive_acc += finetune_correct_all[task_run]
    else:
        finetune_model_syllogisms_acc += finetune_correct_all[task_run]
finetune_model_acc = finetune_model_syllogisms_acc + finetune_model_transitive_acc
logging.info('Prev model accuracy overall is ' + str(average_list(model_acc)) + ', syllogisms is ' + str(average_list(model_acc[:36])) + ', transitive is ' + str(average_list(model_acc[36:])))
logging.info('Finetune model accuracy overall is ' + str(average_list(finetune_model_acc)) + ', syllogisms is ' + str(average_list(finetune_model_syllogisms_acc)) + ', transitive is ' + str(average_list(finetune_model_transitive_acc)))
logging.info(model_acc)
logging.info(finetune_model_acc)


# test set
if args.test_batch_size == 1:
    acc, syl_acc, tra_acc, correct_all = test_model(model, test_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, parse_model_type, False, False, True, return_results=True)
else:
    acc, syl_acc, tra_acc, correct_all = test_model_batch(model, test_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, args.test_batch_size, parse_model_type, False, False, True, return_results=True)

path_results = os.path.join(RESULTS_DIR, f'test_behaviour_results.pkl')
with open(path_results, "wb") as f:
    pickle.dump(correct_all, f)

logging.info('--------------------')
logging.info('Test set')
logging.info('All order overall acc is: {:.3f}, syllogisms acc is: {:.3f}, transitive acc is: {:.3f}'.format(acc, syl_acc, tra_acc))

categories = [
    "2_true_affirm", "2_false_affirm",
    "3_true_affirm", "3_false_affirm",
    "2_true_negate", "2_false_negate",
    "3_true_negate", "3_false_negate"
]
trial_num_per_type = 100
num_order = 6
logging.info('Syllogisms:')
for i in range(len(categories)):
    idx_start = i * trial_num_per_type * num_order
    idx_end = (i + 1) * trial_num_per_type * num_order
    logging.info(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
logging.info('Transitive:')
for i in range(len(categories)):
    idx_start = (i + len(categories)) * trial_num_per_type * num_order
    idx_end = (i + 1 + len(categories)) * trial_num_per_type * num_order
    logging.info(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
