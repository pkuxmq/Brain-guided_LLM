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

def visualize_representations(X, fig_name, labels=None, color_map=None):
    pca = PCA(n_components=2)
    X_ = pca.fit_transform(X)
    df_X = pd.DataFrame(X_, columns=['PCA Dimension 1', 'PCA Dimension 2'])
    df_X['Question Type'] = labels
    df_X['type'] = 'original'
    sorted_labels = sorted(set(labels))
    plt.figure(figsize=(6, 6))
    sns.set(style='ticks', font_scale=0.8)
    sns.scatterplot(data=df_X, x='PCA Dimension 1', y='PCA Dimension 2', hue='Question Type', palette=color_map, s=50, hue_order=sorted_labels)
        
    plt.savefig(fig_name + '.pdf', dpi=600, bbox_inches='tight', format='pdf')
    plt.close()


np.random.seed(0)

parser = argparse.ArgumentParser(description='Visualize model representations')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/visualize/model_rep_structure/')
parser.add_argument('-LLM_PREV_RESULTS_DIR', type=str, default='./results/activations_results/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')

parser.add_argument('-device', type=str, default='cpu')

args = parser.parse_args()

DATA_DIR = args.DATA_DIR
RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

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

# visualize separation between syllogisms and transitive questions
labels = ['syllogisms'] * 36 + ['transitive'] * 34
color_map = {
    'transitive': '#ff7f0e', 
    'syllogisms': '#1f77b4',
}
for layer in layer_index_all:
    X = LLM_prev[:, layer]
    fig_name = os.path.join(RESULTS_DIR, 'visualize_st_layer_' + str(layer))
    visualize_representations(X, fig_name, labels=labels, color_map=color_map)


# visualize structure
# only tran
for layer in layer_index_all:
    X = LLM_prev[:, layer]
    data = X[36:]
    data_ = data - np.mean(data, axis=0, keepdims=True)
    data_ = normalize_vectors(data_)
    cos_sim = data_ @ data_.T
    plt.figure(figsize=(8, 6))
    sns.set(style='ticks', font_scale=1.)
    sns.heatmap(cos_sim, cmap="bwr", xticklabels=[f"{i}" for i in range(cos_sim.shape[0])], yticklabels=[f"{i}" for i in range(cos_sim.shape[0])], cbar=True, vmax=1., vmin=-1)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'structure_layer_'+ str(layer) +'.pdf')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

