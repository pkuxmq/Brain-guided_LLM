import os
import pandas as pd
import pickle
import numpy as np
import scipy.io
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Ridge
import scipy
from scipy.stats import pearsonr
import argparse
import math
from utils import *
import logging

parser = argparse.ArgumentParser(description='Make analysis')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/')
parser.add_argument('-FMRI_RESULTS_DIR', type=str, default='./fmri_data/preprocessed_data_glmsinglesep_newdrroi_topksep/top-10%/')
parser.add_argument('-CEILING_RESULTS_DIR', type=str, default=None)
parser.add_argument('-LLM_RESULTS_DIR', type=str, default='./results/activations_results_localized/')
parser.add_argument('-LLM_rep_type', type=str, default='all', choices=['all', 'hidden', 'attention'])
parser.add_argument('-SCORE_RESULTS_DIR', type=str, default='./results/score_results/')
parser.add_argument('-model_type', type=str, default='mistral')

parser.add_argument('-analyze_type', type=str, default='deductive_reasoning')
parser.add_argument('-signal_type', type=str, default='beta')
parser.add_argument('-ridge_alpha', type=float, default=100.) # if not use RidgeCV
parser.add_argument('-use_ridgecv', action='store_true')
parser.add_argument('-use_cosine', action='store_true')
parser.add_argument('-n_kfold', type=int, default=5)
parser.add_argument('-n_seed', type=int, default=1)

parser.add_argument('-rand_feature', action='store_true')

parser.add_argument('-only_syllogisms', action='store_true')
parser.add_argument('-only_transitive', action='store_true')

parser.add_argument('-manual_seed', type=int, default=0)

args = parser.parse_args()

np.random.seed(args.manual_seed)

DATA_DIR = args.DATA_DIR
LLM_RESULTS_DIR = os.path.join(args.LLM_RESULTS_DIR, args.model_type)
FMRI_RESULTS_DIR = os.path.join(args.FMRI_RESULTS_DIR, 'all_extracted_beta_' + args.analyze_type)
suffix = args.analyze_type
suffix += '_' + args.LLM_rep_type
if args.only_syllogisms:
    suffix += '_onlysyllogisms'
elif args.only_transitive:
    suffix += '_onlytransitive'
if args.use_ridgecv:
    suffix += '_ridgecv'
else:
    suffix += '_ridgealpha{:d}'.format(int(args.ridge_alpha))
if args.use_cosine:
    suffix += '_usecosine'
if args.CEILING_RESULTS_DIR is not None:
    CEILING_RESULTS_DIR = os.path.join(args.CEILING_RESULTS_DIR, suffix)
    # sub * dim
    ceiling_value = np.load(os.path.join(CEILING_RESULTS_DIR, 'ceiling.npy'))
    suffix += 'withceiling'
else:
    ceiling_value = None

if args.rand_feature:
    suffix += '_rand'
if args.manual_seed != 0:
    suffix += '_seed{:d}'.format(args.manual_seed)
SCORE_RESULTS_DIR = os.path.join(args.SCORE_RESULTS_DIR, args.model_type, suffix)
if not os.path.exists(SCORE_RESULTS_DIR):
    os.makedirs(SCORE_RESULTS_DIR)
log_path = os.path.join(SCORE_RESULTS_DIR, 'analysis.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')


# filter these subjects due to missing / abnormal accuracy results or failure to normally process fMRI
filter_index = ['sub-1010', 'sub-1016', 'sub-1026', 'sub-1031', 'sub-1032', 'sub-1035', 'sub-1021']

task_items, data_dict, sub_list, model_ans_all, label_all, correct_all = load_and_filter_behavior_results(DATA_DIR, LLM_RESULTS_DIR, filter_index)
task_run_list = sorted(task_items.keys())
# load prev LLM features
#state_ah_all = load_LLM_features(LLM_RESULTS_DIR, task_run_list)
# N * n_dim
if args.LLM_rep_type == 'all':
    state_ah_all = np.load(os.path.join(LLM_RESULTS_DIR, 'LLM_rep_localized.npy'))
elif args.LLM_rep_type == 'hidden':
    state_ah_all = np.load(os.path.join(LLM_RESULTS_DIR, 'LLM_rep_localized_hidden.npy'))
elif args.LLM_rep_type == 'attention':
    state_ah_all = np.load(os.path.join(LLM_RESULTS_DIR, 'LLM_rep_localized_attention.npy'))

if args.rand_feature:
    state_ah_all = np.random.rand(*state_ah_all.shape)
if args.only_syllogisms:
    state_ah_all = state_ah_all[:36]
elif args.only_transitive:
    state_ah_all = state_ah_all[36:]
# load all fmri data
fmri_all_sub, task_id = get_all_fmri_data_latest(sub_list, FMRI_RESULTS_DIR)
if args.only_syllogisms:
    fmri_all_sub = fmri_all_sub[:, :36]
    task_id = task_id[:36]
elif args.only_transitive:
    fmri_all_sub = fmri_all_sub[:, 36:]
    task_id = task_id[36:]
    
    
def localized_analysis(sub_list, fmri_all_sub, task_id, args, state_ah_all, ceiling_value=None):
    n_sample = state_ah_all.shape[0]
    n_dim = state_ah_all.shape[1]

    sub_score_layer_all = []
    sub_scores = []
    all_scores = []
    for ind, sub in enumerate(sub_list):
        logging.info('\n')
        logging.info('--------------------')
        logging.info(sub)
        logging.info('--------------------')
    
        fmri_state_all = fmri_all_sub[ind]
        logging.info('fMRI state dim: {:}'.format(fmri_state_all.shape[-1]))
        f_dim = fmri_state_all.shape[1]
    
        n_kfold = args.n_kfold
        n_seed = args.n_seed
        if args.use_cosine:
            # 1 * n_kfold * n_seed
            ah_corr_all = np.zeros((1, n_kfold, n_seed))
        else:
            # fdim * n_kfold * n_seed
            ah_corr_all = np.zeros((f_dim, n_kfold, n_seed))
    
        # repeat n_seed times with different random seeds for KFold
        for seed in range(n_seed):
            # split train and test, repeat n_kfold times
            kf = StratifiedKFold(n_splits=n_kfold, shuffle=True, random_state=seed+1)
            for t, (train_index, test_index) in enumerate(kf.split(fmri_state_all, task_id)):
                x_train, x_test, y_train, y_test = state_ah_all[train_index, :], state_ah_all[test_index, :], fmri_state_all[train_index], fmri_state_all[test_index]
                if args.use_ridgecv:
                    if args.use_cosine:
                        ah_corr_all[:, t, seed] = calculate_cos_cv(x_train, y_train, x_test, y_test)
                    else:
                        ah_corr_all[:, t, seed] = calculate_corr_cv(x_train, y_train, x_test, y_test)
                else:
                    ah_corr_all[:, t, seed] = calculate_corr(x_train, y_train, x_test, y_test, args.ridge_alpha)
        # fdim
        ah_corr = np.mean(ah_corr_all, axis=(1,2))
        # divide ceiling
        if ceiling_value is not None:
            ah_corr = ah_corr / ceiling_value[ind, :]
        all_scores.append(ah_corr)
    
        sub_score = np.median(ah_corr)
        sub_score = float(round(sub_score, 3))
        logging.info('Sub score: ' + str(sub_score))
        sub_scores.append(sub_score)

    logging.info('\n')
    logging.info('--------------------')
    logging.info('Overall')
    logging.info('--------------------')
    logging.info('Sub scores:')
    logging.info(sub_scores)
    # n_sub
    sub_score_all_ = np.stack(sub_scores, axis=0)
    sub_score_mad = mad(sub_score_all_)
    sub_score_all = np.median(sub_score_all_, axis=0)
    logging.info('Overall score is {:.3f}, corresponding mad is {:.3f}'.format(sub_score_all, sub_score_mad))

    # get score for each voxel
    # n_sub *  fdim
    all_scores = np.stack(all_scores, axis=0)
    scores = all_scores
    return sub_score_all, scores, sub_score_mad

score, voxel_scores, corresponding_mad = localized_analysis(sub_list, fmri_all_sub, task_id, args, state_ah_all, ceiling_value)

print('brain score is {:.3f}, corresponding mad is {:.3f}'.format(score, corresponding_mad))
mean_score = np.mean(voxel_scores)
print('mean score is {:.3f}'.format(mean_score))
path_scores = os.path.join(SCORE_RESULTS_DIR, 'scores.npy')
np.save(path_scores, voxel_scores)

