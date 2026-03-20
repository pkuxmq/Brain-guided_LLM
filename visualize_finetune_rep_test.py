import os
import pandas as pd
import pickle
from LM import LM_nnsight
import numpy as np
from utils import load_from_json, get_LLM_rep
import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from sklearn.decomposition import PCA
import gc

def visualize_representations_kernel(X, Y, fig_name, labels=None, color_map=None, x_type=None, y_type=None):
    pca = PCA(n_components=2)
    X_ = pca.fit_transform(X)
    pca = PCA(n_components=2)
    Y_ = pca.fit_transform(Y)
    df_X = pd.DataFrame(X_, columns=['PCA Dimension 1', 'PCA Dimension 2'])
    df_X['label'] = labels
    df_X['type'] = 'Original' if x_type is None else x_type
    df_Y = pd.DataFrame(Y_, columns=['PCA Dimension 1', 'PCA Dimension 2'])
    df_Y['label'] = labels
    df_Y['type'] = 'NARF' if y_type is None else y_type
    sorted_labels = sorted(set(labels))
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.set(style='ticks', font_scale=1.5)

    plt.rcParams['legend.title_fontsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    
    # Plot left subplot
    sns.kdeplot(data=df_X, x='PCA Dimension 1', y='PCA Dimension 2', hue='label', palette=color_map, ax=axes[0], fill=False, hue_order=sorted_labels, alpha=0.9, thresh=0.05, legend=True)
    legend = axes[0].get_legend()
    legend.set_title('Question Type')
    
    # Plot right subplot
    sns.kdeplot(data=df_Y, x='PCA Dimension 1', y='PCA Dimension 2', hue='label', palette=color_map, ax=axes[1], fill=False, hue_order=sorted_labels, alpha=0.9, thresh=0.05, legend=True)
    axes[1].get_legend().remove()  # Remove the default legend

    plt.tight_layout()
    plt.savefig(fig_name + '.pdf', dpi=600, bbox_inches='tight', format='pdf')
    plt.close()


np.random.seed(0)

parser = argparse.ArgumentParser(description='Visualize model representations after fine-tuning')
parser.add_argument('-DATA', type=str, default='./data/deductive_reasoning_data_val_new.json')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/visualize/finetune_model_rep/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/')
parser.add_argument('-model_path_narf', type=str, default='./results/finetune_results/qwen1-5b/narf/model/')
parser.add_argument('-model_path_label', type=str, default='./results/finetune_results/qwen1-5b/label/model/')
parser.add_argument('-model_path_narflabel', type=str, default='./results/finetune_results/qwen1-5b/narflabel/model/')
parser.add_argument('-num_per_type', type=int, default=10)

parser.add_argument('-device', type=str, default='cpu')

args = parser.parse_args()

RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

dataset = load_from_json(args.DATA)
device = args.device

# get model representations
rep_path = os.path.join(RESULTS_DIR, 'rep_' + str(args.num_per_type) + '.pkl')
if os.path.exists(rep_path):
    with open(rep_path, "rb") as f:
        [LLM_prev, LLM_narf, LLM_label, LLM_narflabel, labels] = pickle.load(f)
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
else:
    if args.model_type == 'llama2':
        layer_num = 32
        parse_model_type = 'llama2'
    elif args.model_type == 'mistral':
        layer_num = 32
        parse_model_type = 'default'
    elif args.model_type == 'qwen1-5b':
        layer_num = 28
        parse_model_type = 'qwen'
    elif args.model_type == 'qwen7b':
        layer_num = 28
        parse_model_type = 'qwen'
    elif args.model_type == 'llama3':
        layer_num = 32
        parse_model_type = 'llama3'
    else:
        raise("Error! Unsupported model type")
    layer_index_all = list(range(layer_num//4, layer_num*3//4))

    model_original = LM_nnsight(args.model_path, device, parse_model_type=parse_model_type)
    LLM_prev, labels = get_LLM_rep(dataset, model_original, args.num_per_type)
    model_original = None
    gc.collect()
    torch.cuda.empty_cache()

    model_narf = LM_nnsight(args.model_path_narf, device, parse_model_type=parse_model_type)
    LLM_narf, _ = get_LLM_rep(dataset, model_narf, args.num_per_type)
    model_narf = None
    gc.collect()
    torch.cuda.empty_cache()

    model_label = LM_nnsight(args.model_path_label, device, parse_model_type=parse_model_type)
    LLM_label, _ = get_LLM_rep(dataset, model_label, args.num_per_type)
    model_label = None
    gc.collect()
    torch.cuda.empty_cache()

    model_narflabel = LM_nnsight(args.model_path_narflabel, device, parse_model_type=parse_model_type)
    LLM_narflabel, _ = get_LLM_rep(dataset, model_narflabel, args.num_per_type)
    model_narflabel = None
    gc.collect()
    torch.cuda.empty_cache()

    with open(rep_path, "wb") as f:
        pickle.dump([LLM_prev, LLM_narf, LLM_label, LLM_narflabel, labels], f)


# visualize
color_map = {
    'transitive_affirm_true': '#B4E0C7',
    'transitive_affirm_false': '#FFC3A0',
    'transitive_negate_true': '#88BB92',
    'transitive_negate_false': '#FF6B6B',
    'syllogisms_affirm_true': '#66C2A5',
    'syllogisms_affirm_false': '#FF9F8D',
    'syllogisms_negate_true': '#3A9188',
    'syllogisms_negate_false': '#D1495B',
}

# the last finetuned layer
layer = layer_index_all[-1]
# narf vs. original
if not os.path.exists(os.path.join(RESULTS_DIR, 'narf-original')):
    os.makedirs(os.path.join(RESULTS_DIR, 'narf-original'))
X = LLM_prev[:, layer]
Y = LLM_narf[:, layer]
fig_name = os.path.join(RESULTS_DIR, 'narf-original', 'kernelvisualize_layer_' + str(layer) + '_pca')
visualize_representations_kernel(X, Y, fig_name, labels, color_map)

# narflabel vs. label
if not os.path.exists(os.path.join(RESULTS_DIR, 'narflabel-label')):
    os.makedirs(os.path.join(RESULTS_DIR, 'narflabel-label'))
X = LLM_label[:, layer]
Y = LLM_narflabel[:, layer]
fig_name = os.path.join(RESULTS_DIR, 'narflabel-label', 'kernelvisualize_layer_' + str(layer) + '_pca')
visualize_representations_kernel(X, Y, fig_name, labels, color_map)
