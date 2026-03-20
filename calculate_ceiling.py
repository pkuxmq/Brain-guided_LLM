import os
import pandas as pd
import pickle
import numpy as np
import scipy.io
from sklearn.model_selection import KFold, StratifiedKFold
import scipy
from numpy.random.mtrand import RandomState
import argparse
from scipy.optimize import curve_fit
from utils import *
import itertools

def v(x, v0, tau0):
    return v0 * (1 - np.exp(-x / tau0))


parser = argparse.ArgumentParser(description='Calculate ceiling')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/ceiling_results/glmsingle_top10%/')
parser.add_argument('-FMRI_RESULTS_DIR', type=str, default='./fmri_data/preprocessed_data_glmsinglesep_newdrroi_topksep/top-10%/')
parser.add_argument('-analyze_type', type=str, default='deductive_reasoning')
parser.add_argument('-signal_type', type=str, default='beta')
parser.add_argument('-ridge_alpha', type=float, default=100.)
parser.add_argument('-use_ridgecv', action='store_true')
parser.add_argument('-n_kfold', type=int, default=5)

parser.add_argument('-only_syllogisms', action='store_true')
parser.add_argument('-only_transitive', action='store_true')

args = parser.parse_args()

DATA_DIR = args.DATA_DIR
suffix = args.analyze_type
if args.only_syllogisms:
    suffix += '_onlysyllogisms'
elif args.only_transitive:
    suffix += '_onlytransitive'
if args.use_ridgecv:
    suffix += '_ridgecv'
else:
    suffix += '_ridgealpha{:d}'.format(int(args.ridge_alpha))
RESULTS_DIR = os.path.join(args.RESULTS_DIR, suffix)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

filter_index = ['sub-1010', 'sub-1016', 'sub-1026', 'sub-1031', 'sub-1032', 'sub-1035', 'sub-1021']
task_items, data_dict, sub_list = load_question_sublist(DATA_DIR, filter_index)

FMRI_RESULTS_DIR = os.path.join(args.FMRI_RESULTS_DIR, 'all_extracted_'+args.signal_type+'_'+args.analyze_type)
# sub * questions * dim
all_sub_fmri_state, task_id = get_all_fmri_data_latest(sub_list, FMRI_RESULTS_DIR)
if args.only_syllogisms:
    all_sub_fmri_state = all_sub_fmri_state[:, :36, :]
    task_id = task_id[:36]
elif args.only_transitive:
    all_sub_fmri_state = all_sub_fmri_state[:, 36:, :]
    task_id = task_id[36:]

print('shape of fmri states of all subs:')
print(all_sub_fmri_state.shape)
n_sub = all_sub_fmri_state.shape[0]
n_dim = all_sub_fmri_state.shape[2]

n_kfold = args.n_kfold

# sub * dim * n_subsample
all_scores = np.zeros((n_sub, n_dim, n_sub-1))
for sub_num in range(2, len(all_sub) + 1):
    print('sub_num: ' + str(sub_num))
    # enumerate for each sub
    for sub_ind in range(n_sub):
        remaining_ind = [i for i in range(n_sub) if i != sub_ind]
        all_combinations = list(itertools.combinations(remaining_ind, sub_num - 1))
        # n_combination * dim
        scores_combinations = np.zeros((len(all_combinations), n_dim))
        for i, combination in enumerate(all_combinations):
            combination = list(combination)
            source_state = all_sub_fmri_state[combination].transpose((1,2,0))
            source_state = source_state.reshape(source_state.shape[0], -1)
            target_state = all_sub_fmri_state[sub_ind]

            # dim * n_kfold
            corr_all = np.zeros((target_state.shape[1], n_kfold))
            kf = StratifiedKFold(n_splits=n_kfold, shuffle=True, random_state=1)
            for t, (train_index, test_index) in enumerate(kf.split(target_state, task_id)):
                x_train, x_test, y_train, y_test = source_state[train_index], source_state[test_index], target_state[train_index], target_state[test_index]
                if args.use_ridgecv:
                    corr_all[:, t] = calculate_corr_cv(x_train, y_train, x_test, y_test)
                else:
                    corr_all[:, t] = calculate_corr(x_train, y_train, x_test, y_test, args.ridge_alpha)

            scores_combinations[i, :] = np.mean(corr_all, axis=1)
        all_scores[sub_ind, :, sub_num - 2] = np.mean(scores_combinations, axis=0)

# sub * dim
v_value = np.zeros((n_sub, n_dim))
for i in range(n_sub):
    for j in range(n_dim):
        try:
            params, _ = curve_fit(v, list(range(2, len(all_sub) + 1)), all_scores[i, j, :], bounds=([0, -np.inf], [1, np.inf]))
            v_value[i, j] = params[0]
        except RuntimeError: # optimal parameters not found
            v_value[i, j] = all_scores[i, j, -1]

print('ceiling value is: {:.6f}'.format(np.mean(v_value)))

path_results = os.path.join(RESULTS_DIR, 'ceiling.npy')
np.save(path_results, v_value)
