import os
import pandas as pd
import pickle
from LM import LM_nnsight
import numpy as np
from utils import *
import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from sklearn.decomposition import PCA

def visualize_representations(X, intv_dir, fig_name, labels=None, color_map=None):
    pca = PCA(n_components=2)
    X_ = pca.fit_transform(X)
    # calculate the intervened point
    intv_x = {}
    for q_ind_ in intv_dir.keys():
        intv_x[q_ind_] = pca.transform((X[q_ind] + intv_dir[q_ind_]).reshape(1, -1)).flatten()

    colors = np.array([color_map[label] for label in labels])
    df_X = pd.DataFrame(X_, columns=['PCA Dimension 1', 'PCA Dimension 2'])
    df_X['label'] = labels

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set(style='ticks', font_scale=1.5)
    sorted_labels = sorted(set(labels))
    sns.scatterplot(data=df_X, x='PCA Dimension 1', y='PCA Dimension 2', hue='label', palette=color_map, ax=ax, s=60, hue_order=sorted_labels)
    for q_ind_ in intv_x.keys():
        ax.scatter(intv_x[q_ind_][0], intv_x[q_ind_][1], color=colors[q_ind_], s=100, alpha=1., marker='*')
        ax.plot([X_[q_ind_][0], intv_x[q_ind_][0]], [X_[q_ind_][1], intv_x[q_ind_][1]], linestyle='--', color=colors[q_ind_], alpha=0.3, linewidth=1)
        dx = X_[q_ind_][0] - intv_x[q_ind_][0]
        dy = X_[q_ind_][1] - intv_x[q_ind_][1]
        distance = np.sqrt(dx**2 + dy**2)
        arrow_dist = 0.01
        start_x = intv_x[q_ind_][0] + arrow_dist * dx / distance
        start_y = intv_x[q_ind_][1] + arrow_dist * dy / distance
        ax.annotate('', xy=(intv_x[q_ind_][0], intv_x[q_ind_][1]), xytext=(start_x, start_y), arrowprops=dict(arrowstyle='->', color=colors[q_ind_], alpha=0.6, mutation_scale=20))
    ax.legend(title='Question Type', fontsize=10, title_fontsize=12)
        
    plt.savefig(fig_name + '.pdf', dpi=600, bbox_inches='tight', format='pdf')
    plt.close()


np.random.seed(0)

parser = argparse.ArgumentParser(description='Visualize intervention of representations')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/visualize/intervention_dir_pca/')
parser.add_argument('-LLM_PREV_RESULTS_DIR', type=str, default='./results/activations_results/')
parser.add_argument('-INTERVE_INFO_DIR', type=str, default='./results/intervention_sep_results/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')

parser.add_argument('-device', type=str, default='cpu')

args = parser.parse_args()

DATA_DIR = args.DATA_DIR
RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

suffix = 'deductive_reasoning_cosine_lr0.1-iteration200_projalpha10_middlelayer_ridgecv'
INTERVE_INFO_DIR = os.path.join(args.INTERVE_INFO_DIR, args.model_type, suffix)
path_results = os.path.join(INTERVE_INFO_DIR, 'intervention_info.pkl')
with open(path_results, 'rb') as f:
    intervention_info = pickle.load(f)

if args.model_type == 'llama2':
    layer_num = 32
elif args.model_type == 'mistral':
    layer_num = 32
elif args.model_type == 'qwen1-5b':
    layer_num = 28
elif args.model_type == 'qwen7b':
    layer_num = 28
elif args.model_type == 'llama3':
    layer_num = 32
else:
    raise("Error! Unsupported model type")
layer_index_all = list(range(layer_num//4, layer_num*3//4))

LLM_PREV_RESULTS_DIR = os.path.join(args.LLM_PREV_RESULTS_DIR, args.model_type)

# load questions
with open(os.path.join(DATA_DIR, 'task_items.pkl'), 'rb') as f:
    task_items = pickle.load(f)
task_run_list = sorted(task_items.keys())

# load prev LLM features
LLM_prev = load_LLM_features(LLM_PREV_RESULTS_DIR, task_run_list)

# get direction
intv_dir = {}
for layer in layer_index_all:
    intv_dir[layer] = {}
for q_type in ['Syllogisms', 'Transitive']:
    for q_ind in intervention_info[q_type].keys():
        # use the nearest point
        min_avg_dis = None
        for rep_diff in intervention_info[q_type][q_ind]:
            avg_dis = 0
            for layer in layer_index_all:
                avg_dis += np.linalg.norm(rep_diff[layer])
            avg_dis /= len(layer_index_all)
            if min_avg_dis is None or avg_dis < min_avg_dis:
                min_avg_dis = avg_dis
                rep_diff_nearest = rep_diff
        if q_type == 'Syllogisms':
            q_ind_ = q_ind
        else:
            q_ind_ = q_ind + 36

        for layer in layer_index_all:
            intv_dir[layer][q_ind_] = rep_diff_nearest[layer]


# visualize
# first get labels
labels = []
for task_run in sorted(task_items.keys()):
    df = task_items[task_run]
    for i in range(len(df)):
        if 'Transitive' in task_run and '01' in task_run and (i == 7 or i == 13):
            continue
        else:
            trial_type = df['trial_type'][i]
            if 'Transitive' in task_run:
                if 'true_affirm' in trial_type:
                    labels.append('transitive_affirm_true')
                elif 'false_affirm' in trial_type:
                    labels.append('transitive_affirm_false')
                elif 'true_negate' in trial_type:
                    labels.append('transitive_negate_true')
                else:
                    labels.append('transitive_negate_false')
            else:
                if 'true_affirm' in trial_type:
                    labels.append('syllogism_affirm_true')
                elif 'false_affirm' in trial_type:
                    labels.append('syllogism_affirm_false')
                elif 'true_negate' in trial_type:
                    labels.append('syllogism_negate_true')
                else:
                    labels.append('syllogism_negate_false')
color_map = {
    'transitive_affirm_true': '#B4E0C7', 
    'transitive_affirm_false': '#FFC3A0',
    'transitive_negate_true': '#88BB92',
    'transitive_negate_false': '#FF6B6B', 
    'syllogism_affirm_true': '#66C2A5', 
    'syllogism_affirm_false': '#FF9F8D', 
    'syllogism_negate_true': '#3A9188', 
    'syllogism_negate_false': '#D1495B', 
}

for layer in layer_index_all:
    X = LLM_prev[:, layer]
    fig_name = os.path.join(RESULTS_DIR, 'visualize_layer_' + str(layer))
    visualize_representations(X, intv_dir[layer], fig_name, labels=labels, color_map=color_map)

