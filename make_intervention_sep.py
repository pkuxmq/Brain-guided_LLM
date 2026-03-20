import os
import pandas as pd
import pickle
from LM import LM_nnsight
import numpy as np
from utils import *
import argparse
import logging
import random

parser = argparse.ArgumentParser(description='Make intervention')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/intervention_sep_results/')
parser.add_argument('-LLM_PREV_RESULTS_DIR', type=str, default='./results/activations_results/')
parser.add_argument('-FMRI_RESULTS_DIR', type=str, default='./fmri_data/preprocessed_data_glmsinglesep_newdrroi_topksep/top-10%/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/')

parser.add_argument('-analyze_type', type=str, default='deductive_reasoning')
parser.add_argument('-loss_type', type=str, default='cosine')
parser.add_argument('-ridge_alpha', type=float, default=100.) # if not use RidgeCV
parser.add_argument('-use_ridgecv', action='store_true')

parser.add_argument('-lr', type=float, default=1e-1)
parser.add_argument('-iteration', type=int, default=200) # max iterations
parser.add_argument('-intv_scale', type=float, default=1.)
parser.add_argument('-proj_alpha', type=float, default=10.)

parser.add_argument('-iter_interval', type=int, default=5)
parser.add_argument('-fit_consistent_questions', action='store_true')
parser.add_argument('-random', action='store_true') # use random direction
parser.add_argument('-random_fmri', action='store_true')
parser.add_argument('-with_fmri_mean', action='store_true')
parser.add_argument('-add_intercept', action='store_true')

parser.add_argument('-all_layer', action='store_true') # we use middle layers by default
parser.add_argument('-suffix', type=str, default='')

parser.add_argument('-save_rep', action='store_true')
parser.add_argument('-not_save_diff', action='store_true')
parser.add_argument('-save_index', action='store_true')

parser.add_argument('-manual_seed', type=int, default=0)

parser.add_argument('-high_sub_num', type=int, default=None)
parser.add_argument('-low_sub_num', type=int, default=None)

parser.add_argument('-device', type=str, default='cpu')

args = parser.parse_args()

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
if args.add_intercept:
    suffix += '_addintercept'
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
if args.suffix != '':
    suffix += '_' + args.suffix
if args.manual_seed != 0:
    suffix += '_seed' + str(args.manual_seed)
RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type, suffix)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
log_path = os.path.join(RESULTS_DIR, 'results.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')

device = args.device

max_new_tokens = 1
model_path = args.model_path
if args.model_type == 'llama2':
    model = LM_nnsight(model_path, device)
    layer_num = 32
    max_new_tokens = 2
elif args.model_type in ['mistral', 'llama3', 'phi4-mini']:
    model = LM_nnsight(model_path, device)
    layer_num = 32
elif args.model_type in ['qwen1-5b', 'qwen7b']:
    model = LM_nnsight(model_path, device)
    layer_num = 28
elif args.model_type == 'deepseekqwen1-5b':
    model = LM_nnsight(model_path, device)
    layer_num = 28
    max_new_tokens = 2000
elif args.model_type == 'gemma2_9b':
    model = LM_nnsight(model_path, device)
    layer_num = 42
else:
    raise("Error! Unsupported model type")

if args.all_layer:
    layer_index_all = list(range(0, layer_num))
else: # middle layer
    layer_index_all = list(range(layer_num//4, layer_num*3//4))

logging.info('layer index all:')
logging.info(layer_index_all)

LLM_PREV_RESULTS_DIR = os.path.join(args.LLM_PREV_RESULTS_DIR, args.model_type)
FMRI_RESULTS_DIR = os.path.join(args.FMRI_RESULTS_DIR, 'all_extracted_beta_' + args.analyze_type)

# filter these subjects due to missing / abnormal accuracy results or failure to normally process fMRI
filtered_index = ['sub-1010', 'sub-1016', 'sub-1026', 'sub-1031', 'sub-1032', 'sub-1035', 'sub-1021']
acc_sorted_sub_index = ['sub-1030', 'sub-1027', 'sub-1019', 'sub-1001', 'sub-1017', 'sub-1004', 'sub-1034', 'sub-1014', 'sub-1022', 'sub-1008']

task_items, data_dict, sub_list, model_ans_all, label_all, correct_all = load_and_filter_behavior_results(DATA_DIR, LLM_PREV_RESULTS_DIR, filtered_index)

if args.high_sub_num is not None:
    sub_list = acc_sorted_sub_index[:args.high_sub_num]
if args.low_sub_num is not None:
    sub_list = acc_sorted_sub_index[-args.low_sub_num:]

task_run_list = sorted(task_items.keys())
# get behavior results
model_syllogisms_acc, model_transitive_acc, model_acc, sub_task_acc = get_model_human_behavior(correct_all, data_dict, sub_list, return_model_sep_acc=True)

# load prev LLM features
LLM_rep_all = load_LLM_features(LLM_PREV_RESULTS_DIR, task_run_list)
# load all fmri data
fmri_all_sub, _ = get_all_fmri_data_latest(sub_list, FMRI_RESULTS_DIR)

# model prompt
prompt_syllogisms = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe a series of relationships among monosyllabic pseudowords and adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."
prompt_transitive = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe relationships among imaginary characters with comparative adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."

if args.save_rep:
    save_rep_diff = {'Syllogisms': {}, 'Transitive': {}}
    save_rep_diff_notaccumulate = {'Syllogisms': {}, 'Transitive': {}}
if args.save_index:
    save_success_index = {'Syllogisms': {}, 'Transitive': {}}

all_q_type = ['Syllogisms', 'Transitive']
for q_type in all_q_type:
    logging.info('********************')
    logging.info(q_type)
    logging.info('********************')
    # first check sub-model diff
    logging.info('checking model-human difference')
    # the reverse mapping from questions to sub
    map_q2s = {}
    # get intervention index
    intervention_index = {}
    for sub in sub_list:
        logging.info('--------------------')
        logging.info(sub)
        logging.info('--------------------')
        if q_type == 'Syllogisms':
            sub_model_diff = [i for i, (x, y) in enumerate(zip(model_syllogisms_acc, sub_task_acc[sub]['syllogisms'])) if (not x) and y]
        else:
            sub_model_diff = [i for i, (x, y) in enumerate(zip(model_transitive_acc, sub_task_acc[sub]['transitive'])) if (not x) and y]
        logging.info('model is wrong and human is correct:')
        logging.info(sub_model_diff)
        logging.info(len(sub_model_diff))
        # get intervention index
        intervention_index[sub] = sub_model_diff
        for i, ind in enumerate(intervention_index[sub]):
            if ind not in map_q2s.keys():
                map_q2s[ind] = [{'sub': sub, 'i': i}]
            else:
                map_q2s[ind].append({'sub': sub, 'i': i})
    
    intervention_results = {}
    intervention_iter_results = {}
    percent_all = []
    intv_acc_all = []
    # intervention considering each sub
    for i, sub in enumerate(sub_list):
        logging.info('--------------------')
        logging.info(sub)
        logging.info('--------------------')
    
        intervention_results[sub] = []
        intervention_iter_results[sub] = []
    
        # get intervention info
        intervention_all = {}
        if q_type == 'Syllogisms':
            fmri_state_all = fmri_all_sub[i, :36] # syllogisms
            LLM_rep = LLM_rep_all[:36, :, :]
        else:
            fmri_state_all = fmri_all_sub[i, 36:] # transitive
            LLM_rep = LLM_rep_all[36:, :, :]
        if args.random_fmri:
            if args.with_fmri_mean:
                if q_type == 'Syllogisms':
                    fmri_mean = np.mean(fmri_all_sub[i, :36], axis=0, keepdims=True)
                else:
                    fmri_mean = np.mean(fmri_all_sub[i, 36:], axis=0, keepdims=True)
            fmri_state_all = np.random.randn(*fmri_state_all.shape)
            if args.with_fmri_mean:
                fmri_state_all = fmri_state_all - np.mean(fmri_state_all, axis=0, keepdims=True)
                fmri_state_all = fmri_state_all + fmri_mean
    
        if q_type == 'Syllogisms':
            overall_con = [x == y for x, y in zip(model_syllogisms_acc, sub_task_acc[sub]['syllogisms'])]
        else:
            overall_con = [x == y for x, y in zip(model_transitive_acc, sub_task_acc[sub]['transitive'])]
        if args.fit_consistent_questions:
            index_con = [x for x, v in enumerate(overall_con) if v]
        else:
            index_con = list(range(len(overall_con)))
    
        W_all = {}
        # intercept
        if args.add_intercept:
            b_all = {}
        rep_diff_all = {}
        optimizer_all = {}
        std_all = {}
    
        if args.random:
            rand_intervention_all = {}
            for layer in layer_index_all:
                rand_intervention_all[layer] = np.random.randn(*LLM_rep[:, layer, :].shape)
    
        for layer in layer_index_all:
            if args.use_ridgecv:
                # intercept
                if args.add_intercept:
                    W, b = get_W_info_cv(LLM_rep, fmri_state_all, layer, index_con, loss_type=args.loss_type, return_b=True)
                else:
                    W = get_W_info_cv(LLM_rep, fmri_state_all, layer, index_con, loss_type=args.loss_type)
            else:
                if args.add_intercept:
                    W, b = get_W_info(LLM_rep, fmri_state_all, layer, index_con, loss_type=args.loss_type, ridge_alpha=args.ridge_alpha, return_b=True)
                else:
                    W = get_W_info(LLM_rep, fmri_state_all, layer, index_con, loss_type=args.loss_type, ridge_alpha=args.ridge_alpha)
            W_all[layer] = torch.from_numpy(W).float().to(args.device)
            # intercept
            if args.add_intercept:
                b_all[layer] = torch.from_numpy(b).float().to(args.device)
            std_all[layer] = None
        LLM_rep_ = torch.from_numpy(LLM_rep).to(args.device)
        fmri_state_ = torch.from_numpy(fmri_state_all).float().to(args.device)
        for layer in layer_index_all:
            rep_diff = torch.nn.Parameter(torch.zeros_like(LLM_rep_[:, layer, :]))
            rep_diff_all[layer] = rep_diff
            coef = torch.mean(torch.abs(LLM_rep_[:, layer, :]))
            optimizer = torch.optim.AdamW([rep_diff], lr=args.lr*coef)
            optimizer_all[layer] = optimizer
    
    
        success_intervention = [0] * len(intervention_index[sub])
        success_iter = [None] * len(intervention_index[sub])
        max_iter = [None] * len(intervention_index[sub])
        all_messages = {}
        for ind_search in range(args.iteration // args.iter_interval):
            # for random direction, we search scale from 1 std to alpha std
            if args.random and ind_search >= args.proj_alpha:
                break
            intervene_index_current = [ind for i, ind in enumerate(intervention_index[sub]) if success_intervention[i] == 0]
            if len(intervene_index_current) == 0:
                break
            rep_diff_clone_all = {}
            for layer in layer_index_all:
                # to avoid updating fixed rep_diff, make a clone
                rep_diff_clone = rep_diff_all[layer].data.clone().detach()
                # saved for the condition that exhibit invalid output
                rep_diff_clone_all[layer] = rep_diff_clone.clone()
                for n_iter in range(args.iter_interval):
                    # intercept
                    if args.add_intercept:
                        state_pred = torch.matmul((LLM_rep_[:, layer, :] + rep_diff_all[layer]), W_all[layer].T) + b_all[layer]
                    else:
                        state_pred = torch.matmul((LLM_rep_[:, layer, :] + rep_diff_all[layer]), W_all[layer].T)
                    loss_index = list(set(index_con + intervention_index[sub]))
                    if args.loss_type == 'pearsonr':
                        r = pearsonr_pytorch(state_pred[loss_index], fmri_state_[loss_index])
                        loss = -torch.mean(r)
                    elif args.loss_type == 'mse':
                        mse_loss = torch.nn.MSELoss()
                        loss = mse_loss(state_pred, fmri_state_)
                    elif args.loss_type == 'cosine':
                        state_pred = state_pred / (torch.norm(state_pred, dim=1, keepdim=True) + 1e-6)
                        fmri_state_ = fmri_state_ / (torch.norm(fmri_state_, dim=1, keepdim=True) + 1e-6)
                        cos_sim = torch.sum(state_pred * fmri_state_, dim=1)
                        loss = -torch.mean(cos_sim)
                    else:
                        raise("unsupported loss type")
                    optimizer_all[layer].zero_grad()
                    loss.backward()
                    grad_data = rep_diff_all[layer].grad.data.clone()
                    
                    rep_diff_all[layer].grad.data = torch.zeros_like(grad_data)
                    rep_diff_all[layer].grad.data[intervene_index_current] = grad_data[intervene_index_current]
                    optimizer_all[layer].step()
                    # projection
                    with torch.no_grad():
                        norm = torch.norm(rep_diff_all[layer].data[intervene_index_current], dim=1, keepdim=True)
                        if std_all[layer] is None:
                            mean = torch.mean(LLM_rep_[:, layer, :], dim=0, keepdim=True)
                            rep_ = LLM_rep_[:, layer, :] - mean
                            std = torch.sqrt(torch.mean(torch.norm(rep_, dim=1)**2))
                            std_all[layer] = std.item()
                        else:
                            std = std_all[layer]
                        alpha = args.proj_alpha
                        scale = norm.clone()
                        scale[scale > std * alpha] = std * alpha
                        rep_diff_all[layer].data[intervene_index_current] = rep_diff_all[layer].data[intervene_index_current] / norm * scale
                # only set updating rep_diff
                rep_diff_clone[intervene_index_current] = rep_diff_all[layer].data.clone().detach()[intervene_index_current]
                rep_diff_all[layer].data = rep_diff_clone
    
                intervention_all[layer] = rep_diff_all[layer].data.clone().detach().cpu().numpy()
    
            question_index = 0
            for task_run in sorted(task_items.keys()):
                df = task_items[task_run]
                if 'Transitive' in task_run:
                    prompt = prompt_transitive
                    if q_type == 'Syllogisms':
                        continue
                else:
                    prompt = prompt_syllogisms
                    if q_type == 'Transitive':
                        continue
            
                for i in range(len(df)):
                    messages = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                            {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (df['premise1'][i].strip(), df['premise2'][i].strip(), df['premise3'][i].strip(), df['conclusion'][i].strip())},
                    ]
            
                    if 'Transitive' in task_run and '01' in task_run and (i == 7 or i == 13):
                        continue
    
                    if question_index not in intervene_index_current:
                        question_index += 1
                        continue
    
                    q_ind_ = intervention_index[sub].index(question_index)
                    all_messages[q_ind_] = messages
                    intervention_dict = {}
                    for k in intervention_all.keys():
                        intervention_dict[k] = intervention_all[k][question_index] * args.intv_scale
                        if args.random:
                            alpha = ind_search + 1
                            intervention_dict[k] = rand_intervention_all[k][question_index] / np.linalg.norm(rand_intervention_all[k][question_index]) * std_all[k] * alpha * args.intv_scale
                    if args.model_type in ['deepseekqwen1-5b']:
                        ans = model.intervention_multilayer(messages, intervention_dict, max_new_tokens=max_new_tokens, apply_all_tokens=True)
                        # parse ans
                        idx = ans.rfind('</think>')
                        if idx != -1:
                            ans = ans[idx+len('</think>'):].strip('\n').strip()
                            if args.model_type in ['deepseekqwen1-5b']:
                                idx_t = ans.rfind('True')
                                idx_f = ans.rfind('False')
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
                        ans = model.intervention_multilayer(messages, intervention_dict, max_new_tokens=max_new_tokens)
                        ans = ans.strip().strip('</s>') # for llama2
                    if ans in ['True', 'true']:
                        model_ans = 1
                    elif ans in ['False', 'false']:
                        model_ans = 0
                    else:
                        model_ans = None
                        if max_iter[q_ind_] is None:
                            max_iter[q_ind_] = (ind_search - 1) * args.iter_interval
                        # for this question, trace back to the previous rep_diff
                        for layer in layer_index_all:
                            rep_diff_all[layer].data[q_ind_] = rep_diff_clone_all[layer][q_ind_]
                    trial_type = df['trial_type'][i]
                    if 'true' in trial_type:
                        label = 1
                    else:
                        label = 0
                    if model_ans == label:
                        success_intervention[q_ind_] = 1
                        if success_iter[q_ind_] is None:
                            success_iter[q_ind_] = ind_search * args.iter_interval
                    question_index += 1
        intervention_results[sub] = success_intervention
        intervention_iter_results[sub] = {'success': success_iter, 'max': max_iter}
    
        # analyze results
        logging.info(intervention_results[sub])
        logging.info(intervention_iter_results[sub])
        n_success = sum(intervention_results[sub])
        n_total = len(intervention_results[sub])
        if n_total == 0:
            n_total = 1
        percent = n_success * 1.0 / n_total
        logging.info('{:d} successful intervention over {:d} questions, percent: {:.3f}'.format(n_success, n_total, percent))

        if q_type == 'Syllogisms':
            prev_acc = round(average_list(model_syllogisms_acc), 3)
            intv_acc = round((sum(model_syllogisms_acc) + sum(intervention_results[sub])) / len(model_syllogisms_acc), 3)
        else:
            prev_acc = round(average_list(model_transitive_acc), 3)
            intv_acc = round((sum(model_transitive_acc) + sum(intervention_results[sub])) / len(model_transitive_acc), 3)
        logging.info('prev model acc: {:.3f}, intv model acc: {:.3f}'.format(prev_acc, intv_acc))
        percent_all.append(percent)
        intv_acc_all.append(intv_acc)
        
        # save index
        if args.save_index:
            save_success_index[q_type][sub] = []
            for i in range(len(intervention_results[sub])):
                if intervention_results[sub][i]:
                    save_success_index[q_type][sub].append(intervention_index[sub][i])
    
        # save rep
        if args.save_rep:
            for ii in range(len(success_intervention)):
                if success_intervention[ii]:
                    q_ind = intervention_index[sub][ii]
                    if q_ind not in save_rep_diff[q_type].keys():
                        save_rep_diff[q_type][q_ind] = []
                        save_rep_diff_notaccumulate[q_type][q_ind] = []
                    rep_diff_ = {}
                    intervention_dict = {}
                    for layer in layer_index_all:
                        intervention_dict[layer] = rep_diff_all[layer].data[q_ind].clone().detach().cpu().numpy()

                    if not args.not_save_diff:
                        ans, all_states = model.intervention_multilayer(all_messages[ii], intervention_dict, max_new_tokens=max_new_tokens, get_all_intervention_states=True)
                        for layer in layer_index_all:
                            rep_diff_[layer] = all_states[layer] - LLM_rep[q_ind, layer]
                        save_rep_diff[q_type][q_ind].append(rep_diff_)

                    save_rep_diff_notaccumulate[q_type][q_ind].append(intervention_dict)
    
    
    logging.info('--------------------')
    logging.info('Average')
    logging.info('--------------------')
    avg_percent = average_list(percent_all)
    avg_intv_acc = average_list(intv_acc_all)
    logging.info('prev model acc: {:.3f}, intv model acc: {:.3f}'.format(prev_acc, avg_intv_acc))
    logging.info('average percent of modified answers: {:.3f}'.format(avg_percent))
    
    all_index = []
    for sub in sub_list:
        all_index += intervention_index[sub]
    all_index = sorted(list(set(all_index)))
    logging.info('All indices')
    logging.info(all_index)
    all_index_acc = []
    all_percent = []
    all_percent_correct = []
    for ind in all_index:
        logging.info('Question {:d}:'.format(ind))
        ind_sub_info_list = map_q2s[ind]
        n_sub = len(ind_sub_info_list)
        sum_acc = 0
        for i in range(n_sub):
            acc = intervention_results[ind_sub_info_list[i]['sub']][ind_sub_info_list[i]['i']]
            logging.info('{}: {:d}'.format(ind_sub_info_list[i]['sub'], acc))
            sum_acc += acc
        logging.info('In total: {}, {:d}/{:d}'.format(sum_acc>0, sum_acc, n_sub))
        all_index_acc.append(sum_acc > 0)
        all_percent.append(sum_acc * 1. / n_sub)
        if sum_acc > 0:
            all_percent_correct.append(sum_acc * 1. / n_sub)
    logging.info('In sum, {:d}/{:d}'.format(sum(all_index_acc), len(all_index_acc)))
    logging.info('Average percent: {:.3f}'.format(average_list(all_percent)))
    if sum(all_index_acc) > 0:
        logging.info('Average percent for correct: {:.3f}'.format(average_list(all_percent_correct)))
    else:
        logging.info('Average percent for correct: N/A')
        
    if q_type == 'Syllogisms':
        syllogisms_all_index_acc = all_index_acc
        syllogisms_all_percent = all_percent
        syllogisms_all_percent_correct = all_percent_correct
        syllogisms_avg_intv_acc = avg_intv_acc
    else:
        transitive_all_index_acc = all_index_acc
        transitive_all_percent = all_percent
        transitive_all_percent_correct = all_percent_correct
        transitive_avg_intv_acc = avg_intv_acc


print(args.model_type)
print(suffix)

logging.info('In summary, syllogisms: {:d}/{:d}, transitive: {:d}/{:d}'.format(sum(syllogisms_all_index_acc), len(syllogisms_all_index_acc), sum(transitive_all_index_acc), len(transitive_all_index_acc)))
logging.info('Average percent, syllogisms: {:.3f}, transitive: {:.3f}'.format(average_list(syllogisms_all_percent), average_list(transitive_all_percent)))
logging.info('Average percent correct, syllogisms: {:.3f}, transitive: {:.3f}'.format(average_list(syllogisms_all_percent_correct), average_list(transitive_all_percent_correct)))
logging.info('Average intv acc, syllogisms: {:.3f}, transitive: {:.3f}'.format(syllogisms_avg_intv_acc, transitive_avg_intv_acc))

print('In summary, syllogisms: {:d}/{:d}, transitive: {:d}/{:d}'.format(sum(syllogisms_all_index_acc), len(syllogisms_all_index_acc), sum(transitive_all_index_acc), len(transitive_all_index_acc)))
print('Average percent, syllogisms: {:.3f}, transitive: {:.3f}'.format(average_list(syllogisms_all_percent), average_list(transitive_all_percent)))
print('Average percent correct, syllogisms: {:.3f}, transitive: {:.3f}'.format(average_list(syllogisms_all_percent_correct), average_list(transitive_all_percent_correct)))
print('Average intv acc, syllogisms: {:.3f}, transitive: {:.3f}'.format(syllogisms_avg_intv_acc, transitive_avg_intv_acc))



if args.save_rep:
    if not args.not_save_diff:
        path_results = os.path.join(RESULTS_DIR, f'intervention_info.pkl')
        with open(path_results, "wb") as f:
            pickle.dump(save_rep_diff, f)

    path_results = os.path.join(RESULTS_DIR, f'intervention_info_dir.pkl')
    with open(path_results, "wb") as f:
        pickle.dump(save_rep_diff_notaccumulate, f)

if args.save_index:
    path_results = os.path.join(RESULTS_DIR, f'intervention_index_info.pkl')
    with open(path_results, "wb") as f:
        pickle.dump(save_success_index, f)