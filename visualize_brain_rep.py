import os
import pandas as pd
import pickle
import numpy as np
from utils import *
import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from sklearn.decomposition import PCA

def visualize_representations(fmri_all_sub, sub_list, fig_name, labels=None, color_map=None):
    sorted_labels = sorted(set(labels))
    for i in range(fmri_all_sub.shape[0]):
        X = fmri_all_sub[i]
        pca = PCA(n_components=2)
        X_ = pca.fit_transform(X)
        df_X = pd.DataFrame(X_, columns=['PCA Dimension 1', 'PCA Dimension 2'])
        df_X['Question Type'] = labels
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=df_X, x='PCA Dimension 1', y='PCA Dimension 2', hue='Question Type', palette=color_map, s=50, hue_order=sorted_labels)
        
        plt.savefig(fig_name + '_' + sub_list[i] + '.pdf', dpi=600, bbox_inches='tight', format='pdf')
        plt.close()

np.random.seed(0)

parser = argparse.ArgumentParser(description='Visualize representation structure')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/visualize/brain_rep_structure/')
parser.add_argument('-FMRI_RESULTS_DIR', type=str, default='./fmri_data/preprocessed_data_glmsinglesep_newdrroi_topksep/top-10%/')
parser.add_argument('-analyze_type', type=str, default='deductive_reasoning')
parser.add_argument('-signal_type', type=str, default='beta')

parser.add_argument('-device', type=str, default='cpu')

args = parser.parse_args()

DATA_DIR = args.DATA_DIR
RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.analyze_type)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

filter_index = ['sub-1010', 'sub-1016', 'sub-1026', 'sub-1031', 'sub-1032', 'sub-1035', 'sub-1021']
task_items, data_dict, sub_list = load_question_sublist(DATA_DIR, filter_index)

# fmri directory
FMRI_RESULTS_DIR = os.path.join(args.FMRI_RESULTS_DIR, 'all_extracted_'+args.signal_type+'_'+args.analyze_type)
# 10 * 70 * fdim
fmri_all_sub, _ = get_all_fmri_data_latest(sub_list, FMRI_RESULTS_DIR)

# visualize separation between syllogisms and transitive questions
labels = ['syllogisms'] * 36 + ['transitive'] * 34
color_map = {
    'transitive': '#ff7f0e', 
    'syllogisms': '#1f77b4',
}

fig_name = os.path.join(RESULTS_DIR, 'visualize_st_' + args.analyze_type)
visualize_representations(fmri_all_sub, sub_list, fig_name, labels=labels, color_map=color_map)


# visualize structure
# only tran
fmri_all_sub = fmri_all_sub[:, 36:, :]
for i in range(fmri_all_sub.shape[0]):
    data = fmri_all_sub[i]
    data_ = data - np.mean(data, axis=0, keepdims=True)
    data_ = normalize_vectors(data_)
    cos_sim = data_ @ data_.T
    #cos_sim = cos_sim / np.max(np.abs(cos_sim))
    plt.figure(figsize=(6, 6))
    sns.set(style='ticks', font_scale=1.)
    sns.heatmap(cos_sim, cmap="bwr", xticklabels=[f"{i}" for i in range(cos_sim.shape[0])], yticklabels=[f"{i}" for i in range(cos_sim.shape[0])], cbar=True, vmax=1., vmin=-1)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'structure_' + args.analyze_type + '_' + sub_list[i] +'.pdf')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight', format='pdf')
    plt.close()