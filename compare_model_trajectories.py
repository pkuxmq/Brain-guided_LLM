import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import argparse
import json
import gc
import torch
from utils import load_from_json, get_LLM_rep

def plot_trajectory_with_gradient(ax, x, y, z=None, color='blue', base_alpha=0.3, 
                                 final_alpha=0.8, linewidth=1.0, linestyle='-'):
    """
    Plot trajectory with gradient transparency from start to end
    """
    if z is None:  # 2D case
        for i in range(len(x) - 1):
            alpha = base_alpha + (final_alpha - base_alpha) * (i / (len(x) - 1))
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], 
                   color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
    else:  # 3D case
        for i in range(len(x) - 1):
            alpha = base_alpha + (final_alpha - base_alpha) * (i / (len(x) - 1))
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]],
                   color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)

def filter_data_by_sample_type(LLM_rep, labels, sample_type='all'):
    """Filter data based on sample type"""
    if sample_type == 'all':
        return LLM_rep, labels
    
    elif sample_type == 'syllogisms':
        # Take samples with syllogisms labels
        indices = [i for i, label in enumerate(labels) if 'syllogisms' in label]
        filtered_LLM_rep = LLM_rep[indices]
        filtered_labels = [labels[i] for i in indices]
        return filtered_LLM_rep, filtered_labels
    
    elif sample_type == 'transitive':
        # Take samples with transitive labels
        indices = [i for i, label in enumerate(labels) if 'transitive' in label]
        filtered_LLM_rep = LLM_rep[indices]
        filtered_labels = [labels[i] for i in indices]
        return filtered_LLM_rep, filtered_labels
    
    else:
        raise ValueError("sample_type must be 'all', 'syllogisms', or 'transitive'")

def fit_pca_and_transform_trajectories(LLM_rep, layer_index_all, n_components=3, pca_model=None):
    """
    Fit PCA on representations and transform trajectories
    If pca_model is provided, use it for transformation instead of fitting new PCA
    """
    if pca_model is None:
        # Prepare data for PCA fitting - collect all representations for selected layers
        all_data = []
        for layer in layer_index_all:
            all_data.append(LLM_rep[:, layer])  # Shape: (n_questions, hidden_dim)
        
        # Concatenate all data
        all_data = np.concatenate(all_data, axis=0)  # Shape: (n_questions * n_layers, hidden_dim)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(all_data)
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        # Use provided PCA model
        pca = pca_model
        print(f"Using provided PCA model for transformation")
    
    # Transform trajectories
    n_questions = LLM_rep.shape[0]
    trajectories_pca = {}
    for q_ind in range(n_questions):
        trajectory = []
        for layer in layer_index_all:
            rep_pca = pca.transform(LLM_rep[q_ind, layer].reshape(1, -1))
            trajectory.append(rep_pca[0])
        trajectories_pca[q_ind] = np.array(trajectory)
    
    return pca, trajectories_pca

def get_axis_limits(trajectories_pca1, trajectories_pca2, n_components=3):
    """Get consistent axis limits for both models"""
    all_points = []
    
    # Collect all points from both models
    for trajectory in trajectories_pca1.values():
        all_points.extend(trajectory)
    for trajectory in trajectories_pca2.values():
        all_points.extend(trajectory)
    
    all_points = np.array(all_points)
    
    if n_components == 2:
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.01
        y_padding = (y_max - y_min) * 0.01
        
        return (x_min - x_padding, x_max + x_padding), (y_min - y_padding, y_max + y_padding)
    
    else:  # 3D
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.01
        y_padding = (y_max - y_min) * 0.01
        z_padding = (z_max - z_min) * 0.01
        
        return (x_min - x_padding, x_max + x_padding), (y_min - y_padding, y_max + y_padding), (z_min - z_padding, z_max + z_padding)

def plot_2d_trajectories_comparison(trajectories_pca1, trajectories_pca2, layer_index_all, labels, 
                                   save_path, model1_name='Model 1', model2_name='Model 2',
                                   plot_mode='separate', gradient_alpha=False):
    """
    Create 2D visualization comparing trajectories from two models
    """
    # Color map for different question types
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
    
    # Get consistent axis limits
    x_lim, y_lim = get_axis_limits(trajectories_pca1, trajectories_pca2, n_components=2)
    
    if plot_mode == 'combined':
        # Plot both models in one figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot model 1 trajectories
        for q_ind, trajectory in trajectories_pca1.items():
            x, y = trajectory[:, 0], trajectory[:, 1]
            color = color_map[labels[q_ind]]
            
            if gradient_alpha:
                plot_trajectory_with_gradient(ax, x, y, color=color, base_alpha=0.05, 
                                            final_alpha=0.4, linewidth=1.0)
            else:
                ax.plot(x, y, color=color, alpha=0.4, linewidth=1.0)
            
            # Mark points
            ax.scatter(x[0], y[0], color=color, s=30, marker='o', alpha=0.8)
            ax.scatter(x[-1], y[-1], color=color, s=30, marker='s', alpha=0.8)
            if len(x) > 2:
                ax.scatter(x[1:-1], y[1:-1], color=color, s=10, marker='.', alpha=0.6)
        
        # Plot model 2 trajectories with different style
        for q_ind, trajectory in trajectories_pca2.items():
            x, y = trajectory[:, 0], trajectory[:, 1]
            color = color_map[labels[q_ind]]
            
            if gradient_alpha:
                plot_trajectory_with_gradient(ax, x, y, color=color, base_alpha=0.1, 
                                            final_alpha=0.6, linewidth=1.0, linestyle='--')
            else:
                ax.plot(x, y, color=color, alpha=0.6, linewidth=1.0, linestyle='--')
            
            # Mark points
            ax.scatter(x[0], y[0], color=color, s=30, marker='o', alpha=0.8)
            ax.scatter(x[-1], y[-1], color=color, s=40, marker='*', alpha=0.8)
            if len(x) > 2:
                ax.scatter(x[1:-1], y[1:-1], color=color, s=10, marker='.', alpha=0.6)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Create legend
        legend_elements = []
        sorted_labels = sorted(set(labels))
        for label_type in sorted_labels:
            color = color_map[label_type]
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=label_type))
        legend_elements.append(plt.Line2D([0], [0], color='black', lw=1.0, alpha=0.4, label=model1_name))
        legend_elements.append(plt.Line2D([0], [0], color='black', lw=1.0, alpha=0.6, linestyle='--', label=model2_name))
        
        ax.legend(handles=legend_elements, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path + '.pdf', dpi=600, bbox_inches='tight', format='pdf')
        plt.close()
    
    else:  # separate plots
        # Plot 1: Model 1
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for q_ind, trajectory in trajectories_pca1.items():
            x, y = trajectory[:, 0], trajectory[:, 1]
            color = color_map[labels[q_ind]]
            
            if gradient_alpha:
                plot_trajectory_with_gradient(ax, x, y, color=color, base_alpha=0.05, 
                                            final_alpha=0.4, linewidth=1.0)
            else:
                ax.plot(x, y, color=color, alpha=0.4, linewidth=1.0)
            
            ax.scatter(x[0], y[0], color=color, s=30, marker='o', alpha=0.8)
            ax.scatter(x[-1], y[-1], color=color, s=30, marker='s', alpha=0.8)
            if len(x) > 2:
                ax.scatter(x[1:-1], y[1:-1], color=color, s=10, marker='.', alpha=0.6)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}_{model1_name.lower().replace(" ", "_")}.pdf', dpi=600, bbox_inches='tight', format='pdf')
        plt.close()
        
        # Plot 2: Model 2
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for q_ind, trajectory in trajectories_pca2.items():
            x, y = trajectory[:, 0], trajectory[:, 1]
            color = color_map[labels[q_ind]]
            
            if gradient_alpha:
                plot_trajectory_with_gradient(ax, x, y, color=color, base_alpha=0.05, 
                                            final_alpha=0.4, linewidth=1.0)
            else:
                ax.plot(x, y, color=color, alpha=0.4, linewidth=1.0)
            
            ax.scatter(x[0], y[0], color=color, s=30, marker='o', alpha=0.8)
            ax.scatter(x[-1], y[-1], color=color, s=30, marker='s', alpha=0.8)
            if len(x) > 2:
                ax.scatter(x[1:-1], y[1:-1], color=color, s=10, marker='.', alpha=0.6)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}_{model2_name.lower().replace(" ", "_")}.pdf', dpi=600, bbox_inches='tight', format='pdf')
        plt.close()

def plot_3d_trajectories_comparison(trajectories_pca1, trajectories_pca2, layer_index_all, labels, 
                                   save_path, model1_name='Model 1', model2_name='Model 2',
                                   plot_mode='separate', gradient_alpha=False):
    """
    Create 3D visualization comparing trajectories from two models
    """
    # Color map for different question types
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
    
    # Get consistent axis limits
    x_lim, y_lim, z_lim = get_axis_limits(trajectories_pca1, trajectories_pca2, n_components=3)
    
    if plot_mode == 'combined':
        # Plot both models in one figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot model 1 trajectories
        for q_ind, trajectory in trajectories_pca1.items():
            x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
            color = color_map[labels[q_ind]]
            
            if gradient_alpha:
                plot_trajectory_with_gradient(ax, x, y, z, color=color, base_alpha=0.05, 
                                            final_alpha=0.4, linewidth=1.0)
            else:
                ax.plot(x, y, z, color=color, alpha=0.4, linewidth=1.0)
            
            # Mark points
            ax.scatter(x[0], y[0], z[0], color=color, s=25, marker='o', alpha=0.8)
            ax.scatter(x[-1], y[-1], z[-1], color=color, s=25, marker='s', alpha=0.8)
            if len(x) > 2:
                ax.scatter(x[1:-1], y[1:-1], z[1:-1], color=color, s=8, marker='.', alpha=0.6)
        
        # Plot model 2 trajectories with different style
        for q_ind, trajectory in trajectories_pca2.items():
            x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
            color = color_map[labels[q_ind]]
            
            if gradient_alpha:
                plot_trajectory_with_gradient(ax, x, y, z, color=color, base_alpha=0.1, 
                                            final_alpha=0.6, linewidth=1.0, linestyle='--')
            else:
                ax.plot(x, y, z, color=color, alpha=0.6, linewidth=1.0, linestyle='--')
            
            # Mark points
            ax.scatter(x[0], y[0], z[0], color=color, s=25, marker='o', alpha=0.8)
            ax.scatter(x[-1], y[-1], z[-1], color=color, s=35, marker='*', alpha=0.8)
            if len(x) > 2:
                ax.scatter(x[1:-1], y[1:-1], z[1:-1], color=color, s=8, marker='.', alpha=0.6)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
        #ax.view_init(elev=45, azim=45)
        
        # Create legend
        legend_elements = []
        sorted_labels = sorted(set(labels))
        for label_type in sorted_labels:
            color = color_map[label_type]
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=label_type))
        legend_elements.append(plt.Line2D([0], [0], color='black', lw=1.0, alpha=0.4, label=model1_name))
        legend_elements.append(plt.Line2D([0], [0], color='black', lw=1.0, alpha=0.6, linestyle='--', label=model2_name))
        
        ax.legend(handles=legend_elements, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path + '.pdf', dpi=600, bbox_inches='tight', format='pdf')
        plt.close()
    
    else:  # separate plots
        # Plot 1: Model 1
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        for q_ind, trajectory in trajectories_pca1.items():
            x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
            color = color_map[labels[q_ind]]
            
            if gradient_alpha:
                plot_trajectory_with_gradient(ax, x, y, z, color=color, base_alpha=0.05, 
                                            final_alpha=0.4, linewidth=1.0)
            else:
                ax.plot(x, y, z, color=color, alpha=0.4, linewidth=1.0)
            
            ax.scatter(x[0], y[0], z[0], color=color, s=25, marker='o', alpha=0.8)
            ax.scatter(x[-1], y[-1], z[-1], color=color, s=25, marker='s', alpha=0.8)
            if len(x) > 2:
                ax.scatter(x[1:-1], y[1:-1], z[1:-1], color=color, s=8, marker='.', alpha=0.6)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
        #ax.view_init(elev=45, azim=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}_{model1_name.lower().replace(" ", "_")}.pdf', dpi=600, bbox_inches='tight', format='pdf')
        plt.close()
        
        # Plot 2: Model 2
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        for q_ind, trajectory in trajectories_pca2.items():
            x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
            color = color_map[labels[q_ind]]
            
            if gradient_alpha:
                plot_trajectory_with_gradient(ax, x, y, z, color=color, base_alpha=0.05, 
                                            final_alpha=0.4, linewidth=1.0)
            else:
                ax.plot(x, y, z, color=color, alpha=0.4, linewidth=1.0)
            
            ax.scatter(x[0], y[0], z[0], color=color, s=25, marker='o', alpha=0.8)
            ax.scatter(x[-1], y[-1], z[-1], color=color, s=25, marker='s', alpha=0.8)
            if len(x) > 2:
                ax.scatter(x[1:-1], y[1:-1], z[1:-1], color=color, s=8, marker='.', alpha=0.6)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
        #ax.view_init(elev=45, azim=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}_{model2_name.lower().replace(" ", "_")}.pdf', dpi=600, bbox_inches='tight', format='pdf')
        plt.close()

def main():
    """
    Main function to execute the model trajectory comparison
    """
    parser = argparse.ArgumentParser(description='Compare trajectories between two models in PCA space')
    parser.add_argument('-DATA', type=str, default='./data/deductive_reasoning_data_val_new.json')
    parser.add_argument('-RESULTS_DIR', type=str, default='./results/visualize/compare_model_trajectories/')
    parser.add_argument('-model_type', type=str, default='qwen1-5b')
    parser.add_argument('-model_path1', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/', 
                       help='Path to first model')
    parser.add_argument('-model_path2', type=str, default='./latestresults/finetune_results/qwen1-5b/narf/model/', 
                       help='Path to second model')
    parser.add_argument('-model1_name', type=str, default='Original', help='Name for first model')
    parser.add_argument('-model2_name', type=str, default='NARF', help='Name for second model')
    parser.add_argument('-num_per_type', type=int, default=10)
    parser.add_argument('-device', type=str, default='cpu')
    
    # New argument for loading cached representations
    parser.add_argument('-cached_rep_path', type=str, default=None,
                       help='Path to cached representations from analyze_finetune_rep_test_latest.py. If provided, will load from this cache instead of computing representations.')
    parser.add_argument('-model1_type', type=str, choices=['original', 'narf', 'label', 'narflabel'], default='original',
                       help='Which model from cached representations to use as model1 (default: original)')
    parser.add_argument('-model2_type', type=str, choices=['original', 'narf', 'label', 'narflabel'], default='narf',
                       help='Which model from cached representations to use as model2 (default: narf)')
    
    parser.add_argument('-pca_dims', type=int, choices=[2, 3], default=3,
                       help='Number of PCA dimensions for visualization (2 or 3, default: 3)')
    parser.add_argument('-sample_type', type=str, choices=['all', 'syllogisms', 'transitive'], default='all',
                       help='Which samples to include: all, syllogisms, or transitive (default: all)')
    parser.add_argument('-plot_mode', type=str, choices=['combined', 'separate'], default='separate',
                       help='Plot both models in same plot or separate plots (default: separate)')
    parser.add_argument('-gradient_alpha', action='store_true', default=False,
                       help='Use gradient transparency from start to end points to show trajectory direction (default: False)')
    parser.add_argument('-shared_pca', action='store_true', default=False,
                       help='Use model1 PCA space for both models instead of separate PCA spaces (default: False)')
    parser.add_argument('-pca_reference', type=str, choices=['model1', 'original'], default='model1',
                       help='When using shared_pca, which model PCA to use as reference. "original" only works with cached data (default: model1)')

    parser.add_argument('-use_layer_output', action='store_true', default=False,
                       help='Use the output of each layer rather than the attention module (default: False)')
    
    args = parser.parse_args()
    
    RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type)
    if args.use_layer_output:
        RESULTS_DIR = os.path.join(RESULTS_DIR, 'layer_output')
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Set layer configuration based on model type
    parse_model_type = 'default'
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
    elif args.model_type == 'qwen72b':
        layer_num = 80
    elif args.model_type == 'llama3-3_70b':
        layer_num = 80
    elif args.model_type == 'qwen3_4b':
        layer_num = 36
        parse_model_type = 'qwen3'
    elif args.model_type == 'phi4-mini':
        layer_num = 32
    elif args.model_type == 'gemma2_9b':
        layer_num = 42
    else:
        raise ValueError("Error! Unsupported model type")
    
    layer_index_all = list(range(layer_num//4, layer_num*3//4))
    if args.use_layer_output:
        layer_index_all = list(range(layer_num//4+layer_num, layer_num*3//4+layer_num))
    
    # Load data
    print("Loading dataset...")
    dataset = load_from_json(args.DATA)
    
    # Check if we should load from cached representations
    if args.cached_rep_path and os.path.exists(args.cached_rep_path):
        print(f"Loading cached representations from: {args.cached_rep_path}")
        with open(args.cached_rep_path, "rb") as f:
            cached_data = pickle.load(f)
            
        # Extract the specific models based on user selection
        model_map = {
            'original': 0,
            'narf': 1, 
            'label': 2,
            'narflabel': 3
        }
        
        if len(cached_data) >= 5:  # [LLM_prev, LLM_narf, LLM_label, LLM_narflabel, labels]
            LLM_prev, LLM_narf, LLM_label, LLM_narflabel, labels = cached_data
            cached_models = [LLM_prev, LLM_narf, LLM_label, LLM_narflabel]
            
            LLM_rep1 = cached_models[model_map[args.model1_type]]
            LLM_rep2 = cached_models[model_map[args.model2_type]]
            
            print(f"Using {args.model1_type} as model1 and {args.model2_type} as model2")
            print(f"Model1 shape: {LLM_rep1.shape}, Model2 shape: {LLM_rep2.shape}")
        else:
            raise ValueError("Cached representations file does not contain expected data format")
    
    else:
        # Check if our own cached representations exist
        rep_path = os.path.join(RESULTS_DIR, f'rep_comparison_{args.num_per_type}_{args.model1_name}_{args.model2_name}.pkl')
        if os.path.exists(rep_path):
            print("Loading cached representations...")
            with open(rep_path, "rb") as f:
                [LLM_rep1, LLM_rep2, labels] = pickle.load(f)
        else:
            # Import the model wrapper (you'll need to make sure this is available)
            try:
                from LM import LM_nnsight  # Adjust import as needed
            except ImportError:
                print("Error: LM_nnsight module not found. Please ensure it's available.")
                return
            
            print(f"Loading first model: {args.model1_name}...")
            model1 = LM_nnsight(args.model_path1, args.device, parse_model_type=parse_model_type)
            LLM_rep1, labels = get_LLM_rep(dataset, model1, args.num_per_type)
            model1 = None
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"Loading second model: {args.model2_name}...")
            model2 = LM_nnsight(args.model_path2, args.device, parse_model_type=parse_model_type)
            LLM_rep2, _ = get_LLM_rep(dataset, model2, args.num_per_type)
            model2 = None
            gc.collect()
            torch.cuda.empty_cache()
            
            # Cache the representations
            with open(rep_path, "wb") as f:
                pickle.dump([LLM_rep1, LLM_rep2, labels], f)
            print("Representations cached for future use.")
    
    # Filter data based on sample type
    print(f"Filtering data for sample type: {args.sample_type}")
    LLM_rep1_filtered, labels_filtered = filter_data_by_sample_type(LLM_rep1, labels, args.sample_type)
    LLM_rep2_filtered, _ = filter_data_by_sample_type(LLM_rep2, labels, args.sample_type)
    
    print(f"Using {len(labels_filtered)} samples for visualization")
    
    # Fit PCA and transform trajectories
    if args.shared_pca:
        if args.pca_reference == 'original' and args.cached_rep_path and os.path.exists(args.cached_rep_path):
            # Use original model from cache as PCA reference
            print(f"Using shared PCA space - fitting {args.pca_dims}D PCA on original model and applying to both models...")
            
            # Load original model representations if not already model1
            if args.model1_type != 'original':
                # Get original model data from cached data
                with open(args.cached_rep_path, "rb") as f:
                    cached_data = pickle.load(f)
                LLM_original = cached_data[0]  # original is always at index 0
                LLM_original_filtered, _ = filter_data_by_sample_type(LLM_original, labels, args.sample_type)
                
                # Fit PCA on original model
                pca_ref, _ = fit_pca_and_transform_trajectories(
                    LLM_original_filtered, layer_index_all, n_components=args.pca_dims)
            else:
                # Model1 is already original, use it as reference
                pca_ref, trajectories_pca1 = fit_pca_and_transform_trajectories(
                    LLM_rep1_filtered, layer_index_all, n_components=args.pca_dims)
            
            # Apply the reference PCA to both models
            if args.model1_type != 'original':
                pca1, trajectories_pca1 = fit_pca_and_transform_trajectories(
                    LLM_rep1_filtered, layer_index_all, n_components=args.pca_dims, pca_model=pca_ref)
            
            pca2, trajectories_pca2 = fit_pca_and_transform_trajectories(
                LLM_rep2_filtered, layer_index_all, n_components=args.pca_dims, pca_model=pca_ref)
        
        else:
            # Use model1 as PCA reference (original behavior)
            print(f"Using shared PCA space - fitting {args.pca_dims}D PCA on {args.model1_name} and applying to both models...")
            pca1, trajectories_pca1 = fit_pca_and_transform_trajectories(
                LLM_rep1_filtered, layer_index_all, n_components=args.pca_dims)
            
            # Use the same PCA for model 2
            pca2, trajectories_pca2 = fit_pca_and_transform_trajectories(
                LLM_rep2_filtered, layer_index_all, n_components=args.pca_dims, pca_model=pca1)
    else:
        print(f"Using separate PCA spaces for each model...")
        # Fit PCA and transform trajectories for each model separately
        print(f"Fitting {args.pca_dims}D PCA for {args.model1_name}...")
        pca1, trajectories_pca1 = fit_pca_and_transform_trajectories(
            LLM_rep1_filtered, layer_index_all, n_components=args.pca_dims)
        
        print(f"Fitting {args.pca_dims}D PCA for {args.model2_name}...")
        pca2, trajectories_pca2 = fit_pca_and_transform_trajectories(
            LLM_rep2_filtered, layer_index_all, n_components=args.pca_dims)
    
    # Choose visualization based on PCA dimensions
    if args.pca_dims == 2:
        print(f"Creating 2D trajectory comparison ({args.plot_mode} mode)...")
        if args.shared_pca:
            pca_suffix = f"_shared_pca_{args.pca_reference}"
        else:
            pca_suffix = "_separate_pca"
        save_path = os.path.join(RESULTS_DIR, f'trajectory_2d_comparison_{args.sample_type}_{args.plot_mode}{pca_suffix}')
        plot_2d_trajectories_comparison(trajectories_pca1, trajectories_pca2, layer_index_all, 
                                       labels_filtered, save_path, args.model1_name, args.model2_name,
                                       args.plot_mode, args.gradient_alpha)
    else:  # 3D
        print(f"Creating 3D trajectory comparison ({args.plot_mode} mode)...")
        if args.shared_pca:
            pca_suffix = f"_shared_pca_{args.pca_reference}"
        else:
            pca_suffix = "_separate_pca"
        save_path = os.path.join(RESULTS_DIR, f'trajectory_3d_comparison_{args.sample_type}_{args.plot_mode}{pca_suffix}')
        plot_3d_trajectories_comparison(trajectories_pca1, trajectories_pca2, layer_index_all, 
                                       labels_filtered, save_path, args.model1_name, args.model2_name,
                                       args.plot_mode, args.gradient_alpha)
    
    if args.plot_mode == 'separate':
        print(f"Visualizations saved to: {save_path}_{args.model1_name.lower().replace(' ', '_')}.pdf and {save_path}_{args.model2_name.lower().replace(' ', '_')}.pdf")
    else:
        print(f"Visualization saved to: {save_path}.pdf")
    
    print("Model trajectory comparison complete!")

if __name__ == "__main__":
    main() 
