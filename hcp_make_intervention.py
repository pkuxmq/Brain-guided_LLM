import os
import pandas as pd
import pickle
from LM import LM_nnsight
import numpy as np
from hcp_utils import *
import argparse
import logging
import random

parser = argparse.ArgumentParser(description='Make intervention')
parser.add_argument('-RESULTS_DIR', type=str, default='./hcp_results/intervention_results/')
parser.add_argument('-LLM_PREV_RESULTS_DIR', type=str, default='./hcp_results/activations_results/')
parser.add_argument('-FMRI_RESULTS_DIR', type=str, default='../hcp_fmri_results/stimdurMeanRT/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/')

parser.add_argument('-analyze_type', type=str, default='md')
parser.add_argument('-loss_type', type=str, default='cosine')
parser.add_argument('-ridge_alpha', type=float, default=100.) # if not use RidgeCV
parser.add_argument('-use_ridgecv', action='store_true')

parser.add_argument('-lr', type=float, default=1e-1)
parser.add_argument('-iteration', type=int, default=50) # max iterations
parser.add_argument('-intv_scale', type=float, default=1.)
parser.add_argument('-proj_alpha', type=float, default=3.)

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

parser.add_argument('-device', type=str, default='cpu')
parser.add_argument('-bf16', action='store_true') # Added bf16 support

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
else: # middle layer
    layer_index_all = list(range(layer_num//4, layer_num*3//4))

logging.info('layer index all:')
logging.info(layer_index_all)


LLM_PREV_RESULTS_DIR = os.path.join(args.LLM_PREV_RESULTS_DIR, args.model_type)
FMRI_RESULTS_DIR = os.path.join(args.FMRI_RESULTS_DIR)

# Load LLM Global Data
llm_activations_dict, llm_behavior_dict = load_relational_llm_data(LLM_PREV_RESULTS_DIR)
llm_stim_index_map = get_relational_llm_stimulus_index_map()

# Get List of Subjects from fMRI directory
subject_info_list = get_relational_fmri_subjects(FMRI_RESULTS_DIR)

if len(subject_info_list) == 0:
    raise ValueError("No subjects found. Check FMRI_RESULTS_DIR path.")


# Initialize Saving Dictionaries
if args.save_rep:
    # Subject-based storage (New names)
    save_rep_diff_bysubject = {} 
    save_rep_diff_dir_bysubject = {} 
    
    # Question-based aggregation (Original names/structure)
    agg_rep_diff = {'Relational': {}}
    agg_rep_diff_dir = {'Relational': {}}

if args.save_index:
    # Original structure: {'Relational': {sub: []}}
    save_success_index = {'Relational': {}}

all_q_type = ['Relational'] # Unified type

logging.info('********************')
logging.info('Relational Task Intervention')
logging.info('********************')

percent_all = []
intv_acc_all = []
human_acc_all = []

# Map: StimulusName -> List of {'sub': subject_id, 'success': 0/1}
per_question_stats = {}

for sub_info in subject_info_list:
    group = sub_info['group']
    sub = sub_info['subject']
    
    logging.info('--------------------')
    logging.info(f"Group: {group}, Subject: {sub}")
    logging.info('--------------------')

    # Load and Align Data
    data_pack = align_subject_data(sub_info, llm_activations_dict, llm_behavior_dict, llm_stim_index_map)
    if data_pack is None:
        logging.info(f"Skipping {sub}: Data alignment failed (no matching stimuli or Load Error).")
        continue

    # Unpack
    LLM_rep_sub = data_pack['llm_rep']   # (N, Layers, Dim)
    fmri_state_sub = data_pack['fmri_data'] # (N, Voxels)
    model_acc = data_pack['model_acc']
    human_acc = data_pack['human_acc']
    llm_question_indices = data_pack.get('llm_question_indices')

    if args.random_fmri:
        if args.with_fmri_mean:
             fmri_mean = np.mean(fmri_state_sub, axis=0, keepdims=True)
        
        fmri_state_sub = np.random.randn(*fmri_state_sub.shape)
        
        if args.with_fmri_mean:
            fmri_state_sub = fmri_state_sub - np.mean(fmri_state_sub, axis=0, keepdims=True)
            fmri_state_sub = fmri_state_sub + fmri_mean

    # Define Indices
    # 1. Fit Indices
    if args.fit_consistent_questions:
        # Consistent means Model Accuracy matches Human Accuracy (Both Correct or Both Wrong)
        # model_acc and human_acc are numpy arrays of 0/1, so this checks element-wise equality
        consistent_mask = (model_acc == human_acc)
        index_con = np.where(consistent_mask)[0].tolist()
        logging.info(f"Using {len(index_con)} consistent trials for training W")
    else:
        index_con = list(range(len(model_acc)))
    
    # 2. Intervention Indices: Human Correct (1) AND Model Wrong (0)
    # Using boolean mask logic
    mask_intervene = (human_acc == 1) & (model_acc == 0)
    intervention_index = np.where(mask_intervene)[0].tolist()
    
    logging.info(f"Total Trials: {len(model_acc)}")
    logging.info(f"Intervention Targets (Human=1, Model=0): {len(intervention_index)} trials")
    logging.info(f"Indices: {intervention_index}")
    if llm_question_indices is not None:
        llm_intervention_index = [llm_question_indices[i] for i in intervention_index]
        logging.info(f"LLM 96-question Indices: {llm_intervention_index}")
    
    if len(intervention_index) == 0:
        logging.info("No intervention targets found. Skipping subject.")
        continue

    # Process per layer
    W_all = {}
    if args.add_intercept:
        b_all = {}
    rep_diff_all = {}
    optimizer_all = {}
    std_all = {}

    rand_intervention_all = {}  # Store random vectors if args.random is true
    if args.random:
        # Generate random vectors for all layers once
        for layer in layer_index_all:
             # Shape: (N_samples, Dim) - same as LLM_rep
             rand_intervention_all[layer] = np.random.randn(*LLM_rep_sub[:, layer, :].shape)

    for layer in layer_index_all:
        if args.use_ridgecv:
            if args.add_intercept:
                W, b = get_W_info_cv(LLM_rep_sub, fmri_state_sub, layer, index_con, loss_type=args.loss_type, return_b=True)
            else:
                W = get_W_info_cv(LLM_rep_sub, fmri_state_sub, layer, index_con, loss_type=args.loss_type)
        else:
             if args.add_intercept:
                W, b = get_W_info(LLM_rep_sub, fmri_state_sub, layer, index_con, loss_type=args.loss_type, ridge_alpha=args.ridge_alpha, return_b=True)
             else:
                W = get_W_info(LLM_rep_sub, fmri_state_sub, layer, index_con, loss_type=args.loss_type, ridge_alpha=args.ridge_alpha)
        
        W_all[layer] = torch.from_numpy(W).float().to(args.device)
        if args.add_intercept:
            b_all[layer] = torch.from_numpy(b).float().to(args.device)
        std_all[layer] = None

    # Determine Device Tensors - TORCH PHASE
    LLM_rep_ = torch.from_numpy(LLM_rep_sub).to(args.device)
    fmri_state_ = torch.from_numpy(fmri_state_sub).float().to(args.device)

    # Init Optimization
    for layer in layer_index_all:
        rep_diff = torch.nn.Parameter(torch.zeros_like(LLM_rep_[:, layer, :]))
        rep_diff_all[layer] = rep_diff
        coef = torch.mean(torch.abs(LLM_rep_[:, layer, :]))
        optimizer = torch.optim.AdamW([rep_diff], lr=args.lr*coef)
        optimizer_all[layer] = optimizer

    # Run Intervention Optimization
    
    success_intervention = [0] * len(intervention_index)
    success_iter = [None] * len(intervention_index)
    max_iter = [None] * len(intervention_index)
    
    # Pre-construct messages for validation for all intervention targets
    all_messages_for_targets = []
    target_stimuli_info_debug = []
    
    for relative_idx in intervention_index:
        # data_pack now contains the 'stimulus_info' list aligned with the data
        info = data_pack['stimulus_info'][relative_idx]
        
        if info is None:
            logging.error(f"Info is None for index {relative_idx}")
            all_messages_for_targets.append(None)
        else:
            msgs = construct_relational_prompt(info)
            all_messages_for_targets.append(msgs)
            target_stimuli_info_debug.append(info)

    for ind_search in range(args.iteration // args.iter_interval):
        # Identify indices that are NOT YET successful
        # We only define intervene_index_current based on failures.
        
        # for random direction, we search scale from alpha/(args.iteration//args.iter_interval) std to alpha std

        # intervention_index is list of absolute indices in the batch.
        # success_intervention is 0/1 array for these items.
        
        not_success_mask = [val == 0 for val in success_intervention]
        
        if sum(not_success_mask) == 0:
            break
            
        # Get absolute indices for current intervention batch
        intervene_index_current = [intervention_index[i] for i in range(len(intervention_index)) if not_success_mask[i]]
        # Also need relative indices (0..K) to index into `success_intervention`
        intervene_indices_rel = [i for i in range(len(intervention_index)) if not_success_mask[i]]

        rep_diff_clone_all = {}
        intervention_all = {}

        for layer in layer_index_all:
             # Snapshot before update (for fallback/reset if needed, logic from original)
             rep_diff_clone = rep_diff_all[layer].data.clone().detach()
             rep_diff_clone_all[layer] = rep_diff_clone.clone()

             for n_iter in range(args.iter_interval):
                if args.add_intercept:
                    state_pred = torch.matmul((LLM_rep_[:, layer, :] + rep_diff_all[layer]), W_all[layer].T) + b_all[layer]
                else:
                    state_pred = torch.matmul((LLM_rep_[:, layer, :] + rep_diff_all[layer]), W_all[layer].T)
                
                loss_index = list(set(index_con + intervention_index))
                
                if args.loss_type == 'pearsonr':
                    r = pearsonr_pytorch(state_pred[loss_index], fmri_state_[loss_index])
                    loss = -torch.mean(r)
                elif args.loss_type == 'mse':
                     mse_loss = torch.nn.MSELoss()
                     loss = mse_loss(state_pred[loss_index], fmri_state_[loss_index])
                elif args.loss_type == 'cosine':
                    sp = state_pred[loss_index]
                    fs = fmri_state_[loss_index]
                    sp = sp / (torch.norm(sp, dim=1, keepdim=True) + 1e-6)
                    fs = fs / (torch.norm(fs, dim=1, keepdim=True) + 1e-6)
                    cos_sim = torch.sum(sp * fs, dim=1)
                    loss = -torch.mean(cos_sim)
                
                optimizer_all[layer].zero_grad()
                loss.backward()
                
                grad_data = rep_diff_all[layer].grad.data.clone()
                rep_diff_all[layer].grad.data = torch.zeros_like(grad_data)
                rep_diff_all[layer].grad.data[intervene_index_current] = grad_data[intervene_index_current]
                
                optimizer_all[layer].step()

                # Projection / Regularization logic (Standard deviation constraint)
                with torch.no_grad():
                    norm = torch.norm(rep_diff_all[layer].data[intervene_index_current], dim=1, keepdim=True)
                    if std_all[layer] is None:
                        mean = torch.mean(LLM_rep_[:, layer, :], dim=0, keepdim=True)
                        rep_ = LLM_rep_[:, layer, :] - mean
                        # Calculate std of projections on the direction
                        std = torch.sqrt(torch.mean(torch.norm(rep_, dim=1)**2)) # Simple scalar std proxy
                        std_all[layer] = std.item()
                    else:
                        std = std_all[layer]
                        
                    alpha = args.proj_alpha
                    # Limit the magnitude of the perturbation vector
                    scale = norm.clone()
                    thresh = std * alpha
                    # If norm > alpha*std, scale down. 
                    scale[norm > thresh] = thresh
                    
                    # Apply scaling
                    rep_diff_all[layer].data[intervene_index_current] = rep_diff_all[layer].data[intervene_index_current] / (norm + 1e-6) * scale
             
             rep_diff_clone[intervene_index_current] = rep_diff_all[layer].data.clone().detach()[intervene_index_current]
             rep_diff_all[layer].data = rep_diff_clone
             
             intervention_all[layer] = rep_diff_all[layer].data.clone().detach().cpu().numpy()

        # ==========================================
        # VALIDATION STEP: Check if intervention worked
        # ==========================================
        
        for i_rel in intervene_indices_rel:
            q_ind_abs = intervention_index[i_rel]
            msgs = all_messages_for_targets[i_rel]
            
            if msgs is None: continue 
            
            # Construct intervention dict for this specific sample
            # Dictionary Key: Layer -> Value: Vector (1, Dim)
            intervention_dict = {}
            for k in intervention_all.keys():
                # Extract the vector for this specific question index
                # rep_diff_all shape is (N_samples, Dim)
                
                if args.random:
                     # For random direction, we scale from 1 std to proj_alpha std
                     # ind_search (0..N) is the step.
                     
                     alpha = (args.proj_alpha / (args.iteration // args.iter_interval)) * (ind_search + 1)
                     # Only if we found std_all (computed during loop)
                     if std_all[k] is not None:
                         # Get random vector for this item
                         rand_vec = rand_intervention_all[k][q_ind_abs]
                         
                         # Normalize and Scale
                         # Note: rand_vec is numpy
                         norm_rand = np.linalg.norm(rand_vec)
                         if norm_rand > 0:
                             rand_vec = rand_vec / norm_rand
                         
                         vec = rand_vec * std_all[k] * alpha * args.intv_scale
                         intervention_dict[k] = vec
                     else:
                         # Should not happen if std logic ran, but fallback
                         intervention_dict[k] = np.zeros_like(intervention_all[k][q_ind_abs])
                else:
                    # Normal Optimized Intervention
                    vec = intervention_all[k][q_ind_abs]
                    intervention_dict[k] = vec * args.intv_scale 

            # Run Model
            try:
                if 'deepseek' in args.model_type.lower():
                     ans = model.intervention_multilayer(msgs, intervention_dict, max_new_tokens=max_new_tokens, apply_all_tokens=True)
                else:
                     ans = model.intervention_multilayer(msgs, intervention_dict, max_new_tokens=max_new_tokens)
            except Exception as e:
                logging.error(f"Model generation failed for {sub} item {q_ind_abs}: {e}")
                continue

            # Parse Answer
            model_ans_val = parse_relational_response(ans, model_type=args.model_type, warn_ambiguous=False)
            
            # Retrieve Label from Item Info
            stim_name = data_pack['stimuli'][q_ind_abs]
            if 'YES' in stim_name.upper() or 'TRUE' in stim_name.upper():
                label = 1
            elif 'NO' in stim_name.upper() or 'FALSE' in stim_name.upper():
                label = 0
            else:
                label = -1
            
            if model_ans_val == label:
                 success_intervention[i_rel] = 1
                 if success_iter[i_rel] is None:
                     success_iter[i_rel] = ind_search * args.iter_interval
            
            elif model_ans_val == -1: # Invalid / Ambiguous
                 if max_iter[i_rel] is None:
                     max_iter[i_rel] = (ind_search - 1) * args.iter_interval
                 
                 # Revert optimization for this item to previous safe state
                 for layer in layer_index_all:
                      rep_diff_all[layer].data[q_ind_abs] = rep_diff_clone_all[layer][q_ind_abs]

    # Analyze results
    n_success = sum(success_intervention)
    n_total = len(intervention_index)
    if n_total == 0: n_total = 1
    
    percent = n_success / n_total
    
    prev_acc = np.mean(model_acc)
    # success_intervention only counts newly fixed items (indices where model was 0)
    intv_acc = (np.sum(model_acc) + n_success) / len(model_acc)
    human_acc_mean = np.mean(human_acc)
    
    logging.info(f"Subject {sub} Results:")
    logging.info(f"Success Mask: {success_intervention}")
    # Show iteration steps for success (None if failed)
    logging.info(f"Success Iterations: {success_iter}")
    logging.info('{:d} successful intervention over {:d} questions, percent: {:.3f}'.format(n_success, n_total, percent))
    logging.info('human acc: {:.3f}, prev model acc: {:.3f}, intv model acc: {:.3f}'.format(human_acc_mean, prev_acc, intv_acc))
    
    # Update Global Stats Sets & Per-Question Stats
    for i_rel, q_ind_abs in enumerate(intervention_index):
        stim_name = data_pack['stimuli'][q_ind_abs]
        
        is_success = success_intervention[i_rel]
        
        if stim_name not in per_question_stats:
            per_question_stats[stim_name] = []
        per_question_stats[stim_name].append({'sub': sub, 'success': is_success})

    percent_all.append(percent)
    intv_acc_all.append(intv_acc)
    human_acc_all.append(human_acc_mean)

    # Save results for this subject
    if args.save_index:
        successful_trial_indices = []
        successful_stimuli = []
        for i_rel, is_success in enumerate(success_intervention):
            if is_success:
                successful_trial_indices.append(intervention_index[i_rel])
                q_ind_abs = intervention_index[i_rel]
                if q_ind_abs < len(data_pack['stimuli']):
                    successful_stimuli.append(data_pack['stimuli'][q_ind_abs])
        
        # Save to Aggregated Structure (sub-based dict inside 'Relational' key)
        save_success_index['Relational'][sub] = {'indices': successful_trial_indices, 'stimuli': successful_stimuli}

    if args.save_rep:
        if sub not in save_rep_diff_bysubject: save_rep_diff_bysubject[sub] = {}
        if sub not in save_rep_diff_dir_bysubject: save_rep_diff_dir_bysubject[sub] = {}

        # Iterate only through successful interventions to confirm and save
        for i_rel, is_success in enumerate(success_intervention):
            if is_success:
                q_ind_abs = intervention_index[i_rel]
                msgs = all_messages_for_targets[i_rel]
                
                # Get Stimulus Name for Aggregation
                # Stimuli list was stored in data_pack['stimuli']
                stim_name = data_pack['stimuli'][q_ind_abs]
                
                final_intervention_dict = {}
                
                for k in layer_index_all:
                     final_intervention_dict[k] = rep_diff_all[k].data[q_ind_abs].clone().detach().cpu().numpy()

                # 1. Subject-based Saving
                save_rep_diff_dir_bysubject[sub][q_ind_abs] = final_intervention_dict
                
                # 2. Aggregated Saving (Vectors)
                if stim_name not in agg_rep_diff_dir['Relational']:
                    agg_rep_diff_dir['Relational'][stim_name] = []
                agg_rep_diff_dir['Relational'][stim_name].append(final_intervention_dict)
                
                # Effect (Difference in Hidden States)
                if not args.not_save_diff and msgs is not None:
                    # Run model again to get states
                    try:
                        if 'deepseek' in args.model_type.lower():
                            _, all_states = model.intervention_multilayer(msgs, final_intervention_dict, max_new_tokens=1, apply_all_tokens=True, get_all_intervention_states=True)
                        else:
                            _, all_states = model.intervention_multilayer(msgs, final_intervention_dict, max_new_tokens=1, get_all_intervention_states=True)
                        
                        effect_dict = {}
                        for layer in layer_index_all:
                            # State diff = Intervened - Original
                            # Need Original. LLM_rep_sub is in numpy (N, Layers, Dim)
                            original_vec = LLM_rep_sub[q_ind_abs, layer, :]
                            effect_dict[layer] = all_states[layer] - original_vec
                        
                        # 1. Subject-based Saving
                        save_rep_diff_bysubject[sub][q_ind_abs] = effect_dict
                        
                        # 2. Aggregated Saving (Effects)
                        if stim_name not in agg_rep_diff['Relational']:
                            agg_rep_diff['Relational'][stim_name] = []
                        agg_rep_diff['Relational'][stim_name].append(effect_dict)

                    except Exception as e:
                        logging.error(f"Error calculating effect for {sub} {q_ind_abs}: {e}")

# Save to disk
logging.info('--------------------')
logging.info('Average across subjects')
logging.info('--------------------')
if len(percent_all) > 0:
    logging.info('Average percent of modified answers: {:.3f}'.format(np.mean(percent_all)))
    logging.info('Average intv model acc: {:.3f}'.format(np.mean(intv_acc_all)))

    # Detailed Question Statistics
    sorted_stimuli = sorted(list(per_question_stats.keys()))
    logging.info('All indices (Stimuli Names):')
    logging.info(sorted_stimuli)
    
    all_index_acc = []     # Did at least one subject fix this question?
    all_percent = []       # Avg restoration rate for this question across subjects
    all_percent_correct = [] # Avg restoration rate (only including questions that were fixed at least once)
    
    for stim in sorted_stimuli:
        logging.info(f'Question {stim}:')
        stat_list = per_question_stats.get(stim, [])
        n_sub = len(stat_list)
        sum_acc = 0
        for item in stat_list:
            acc = item['success']
            # logging.info(f"{item['sub']}: {acc}") # Optional: less verbose than original
            sum_acc += acc
        
        logging.info('In total: {}, {:d}/{:d}'.format(sum_acc>0, sum_acc, n_sub))
        
        all_index_acc.append(sum_acc > 0)
        all_percent.append(sum_acc * 1.0 / n_sub)
        if sum_acc > 0:
             all_percent_correct.append(sum_acc * 1.0 / n_sub)

    logging.info('In sum, {:d}/{:d}'.format(sum(all_index_acc), len(all_index_acc)))
    print('In sum, {:d}/{:d}'.format(sum(all_index_acc), len(all_index_acc)))
    logging.info('Average percent: {:.3f}'.format(np.mean(all_percent)))
    if len(all_percent_correct) > 0:
        logging.info('Average percent for correct: {:.3f}'.format(np.mean(all_percent_correct)))
    else:
        logging.info('Average percent for correct: N/A')

    if len(percent_all) > 1:
        # Correlation Analysis
        from scipy.stats import pearsonr
        
        # Human Acc vs Intervention Success
        if np.std(human_acc_all) > 0 and np.std(percent_all) > 0:
            corr_h, p_h = pearsonr(human_acc_all, percent_all)
            logging.info(f'Correlation (Human Acc vs Intervention Success): r={corr_h:.3f}, p={p_h:.3f}')
        else:
             logging.info('Correlation (Human Acc vs Intervention Success): Undefined (Zero Variance)')

if args.save_rep:
    if not args.not_save_diff:
        # Save Question-Aggregated Data (Original Names)
        path_results = os.path.join(RESULTS_DIR, f'intervention_info.pkl')
        with open(path_results, "wb") as f:
            pickle.dump(agg_rep_diff, f)

        # Save Subject-Based Data (New Names)
        path_results_sub = os.path.join(RESULTS_DIR, f'intervention_info_bysubject.pkl')
        with open(path_results_sub, "wb") as f:
            pickle.dump(save_rep_diff_bysubject, f)

    # Save Question-Aggregated Vectors (Original Names)
    path_results_dir = os.path.join(RESULTS_DIR, f'intervention_info_dir.pkl')
    with open(path_results_dir, "wb") as f:
        pickle.dump(agg_rep_diff_dir, f)

    # Save Subject-Based Vectors (New Names)
    path_results_dir_sub = os.path.join(RESULTS_DIR, f'intervention_info_dir_bysubject.pkl')
    with open(path_results_dir_sub, "wb") as f:
        pickle.dump(save_rep_diff_dir_bysubject, f)

if args.save_index:
    path_results = os.path.join(RESULTS_DIR, f'intervention_index_info.pkl')
    with open(path_results, "wb") as f:
        pickle.dump(save_success_index, f)

logging.info(f"Saved intervention vectors and effects to {RESULTS_DIR}")

#print("Done.")