import os
import pandas as pd
import pickle
from LM import LM_nnsight_base
import numpy as np
import argparse
from utils import load_LLM_features, load_LLM_features_control
from scipy.stats import ttest_ind #, false_discovery_control

parser = argparse.ArgumentParser(description='Get model activations')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/') # path to get task items
parser.add_argument('-RESULTS_DIR', type=str, default='./results/activations_results_localized/') # path to save results
parser.add_argument('-model_type', type=str, default='llama2_base')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Llama-2-7b-hf/')
parser.add_argument('-device', type=str, default='cpu')

args = parser.parse_args()

DATA_DIR = args.DATA_DIR
LM_model = LM_nnsight_base
RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

model_path = args.model_path
device = args.device

max_new_tokens = 1
model = LM_model(model_path, device)

# get task items
with open(os.path.join(DATA_DIR, 'task_items.pkl'), 'rb') as f:
    task_items = pickle.load(f)

prompt_syllogisms = "You are an expert in performing logical reasoning. Three premises and one conclusion will be prsented each round. The premises describe a series of relationships among monosyllabic pseudowords and adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the answer.\n\n"
prompt_transitive = "You are an expert in performing logical reasoning. Three premises and one conclusion will be presented each round. The premises describe relationships among imaginary characters with comparative adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the answer.\n\n"

prompt_syllogisms_control = "The following will present three premises that describe a series of relationships among monosyllabic pseudowords and adjectives. You should just read the descriptions and do nothing. Only respond 'I have read the premises' after the premises."
prompt_transitive_control = "The following will present three premises that describe relationships among imaginary characters with comparative adjectives. You should just read the descriptions and do nothing. Only respond 'I have read the premises' after the premises."

model_ans_all = {}
label_all = {}
correct_all = {}

print('Getting activations for ' + args.model_type)
print('\n')

for task_run in sorted(task_items.keys()):
    df = task_items[task_run]
    print(task_run)
    model_ans_all[task_run] = []
    label_all[task_run] = []
    correct_all[task_run] = []
    if 'Transitive' in task_run:
        prompt = prompt_transitive
    else:
        prompt = prompt_syllogisms

    all_state_h = []
    all_state_a = []
    for i in range(len(df)):
        messages = prompt + "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s.\nAnswer:\n" % (df['premise1'][i].strip(), df['premise2'][i].strip(), df['premise3'][i].strip(), df['conclusion'][i].strip())

        ans, all_tokens = model(messages, max_new_tokens=max_new_tokens, get_all_tokens=True)
        
        if ans in ['True', 'true']:
            model_ans = 1
        elif ans in ['False', 'false']:
            model_ans = 0
        else:
            model_ans = None
            print('----------')
            print(task_run + ' problem ' + str(i))
            print('No answer. The output is ' + ans)
            print('----------')
        model_ans_all[task_run].append(model_ans)

        trial_type = df['trial_type'][i]
        if 'true' in trial_type:
            label = 1
        else:
            label = 0
        label_all[task_run].append(label)
        correct_all[task_run].append(model_ans==label)

        state_h, state_a = model.get_all_states_with_tokens(tokens=all_tokens[:-1])
        # last token
        state_h = state_h[:,-1]
        state_a = state_a[:,-1]
        all_state_h.append(state_h)
        all_state_a.append(state_a)
    all_state_h = np.stack(all_state_h, axis=0)
    all_state_a = np.stack(all_state_a, axis=0)
    path_h = os.path.join(RESULTS_DIR, f'{task_run}_hidden.npy')
    path_a = os.path.join(RESULTS_DIR, f'{task_run}_attention.npy')
    np.save(path_h, all_state_h)
    np.save(path_a, all_state_a)
    print('--------------------')
    print('Accuracy for ' + task_run + ' is: ' + str(sum(correct_all[task_run]) * 1.0 / len(correct_all[task_run])))
    if 'Transitive' in task_run and '01' in task_run:
        print('Note: Accuracy for ' + task_run + 'may be imprecise')
    print('--------------------')

path_results = os.path.join(RESULTS_DIR, f'behaviour_results.pkl')
    
with open(path_results, "wb") as f:
    pickle.dump([model_ans_all, label_all, correct_all], f)





print('Getting control activations for ' + args.model_type)
print('\n')

for task_run in sorted(task_items.keys()):
    df = task_items[task_run]
    print(task_run)
    if 'Transitive' in task_run:
        prompt = prompt_transitive_control
    else:
        prompt = prompt_syllogisms_control

    all_state_h_control = []
    all_state_a_control = []
    for i in range(len(df)):
        messages = prompt + "Premises:\n1. %s.\n2. %s.\n3. %s.\nResponse:\n" % (df['premise1'][i].strip(), df['premise2'][i].strip(), df['premise3'][i].strip())

        ans, all_tokens = model(messages, max_new_tokens=max_new_tokens, get_all_tokens=True)
        
        state_h, state_a = model.get_all_states_with_tokens(tokens=all_tokens[:-1])
        # last token
        state_h = state_h[:,-1]
        state_a = state_a[:,-1]
        all_state_h_control.append(state_h)
        all_state_a_control.append(state_a)
    all_state_h_control = np.stack(all_state_h_control, axis=0)
    all_state_a_control = np.stack(all_state_a_control, axis=0)
    path_h = os.path.join(RESULTS_DIR, f'{task_run}_hidden_control.npy')
    path_a = os.path.join(RESULTS_DIR, f'{task_run}_attention_control.npy')
    np.save(path_h, all_state_h_control)
    np.save(path_a, all_state_a_control)

# N * (n_layer*2) * n_dim
LLM_rep = load_LLM_features(RESULTS_DIR, sorted(task_items.keys()))
LLM_rep_control = load_LLM_features_control(RESULTS_DIR, sorted(task_items.keys()))
n_layer = LLM_rep.shape[1] // 2

p_values_matrix = np.zeros((LLM_rep.shape[1], LLM_rep.shape[2]))
t_values_matrix = np.zeros((LLM_rep.shape[1], LLM_rep.shape[2]))

for layer_idx in range(LLM_rep.shape[1]):
    pos_act = np.abs(LLM_rep[:, layer_idx, :])
    neg_act = np.abs(LLM_rep_control[:, layer_idx, :])

    t_values_matrix[layer_idx], p_values_matrix[layer_idx] = ttest_ind(pos_act, neg_act, axis=0, equal_var=False)

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

def is_bottomk(a, k=1):
    _, rix = np.unique(a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

#num_units = int(0.01 * LLM_rep.shape[1] * LLM_rep.shape[2])  # top 1%
num_units = LLM_rep.shape[2] # corresponds to one layer size


# setting 1: all attention + layer
mask = is_topk(t_values_matrix, k=num_units)
LLM_rep_localized = LLM_rep[:, mask.astype(bool)]

path_localized = os.path.join(RESULTS_DIR, f'LLM_rep_localized.npy')
np.save(path_localized, LLM_rep_localized)
path_mask = os.path.join(RESULTS_DIR, f'LLM_rep_localized_mask.npy')
np.save(path_mask, mask)


# setting 2: only attention layers
mask = is_topk(t_values_matrix[:n_layer], k=num_units)
LLM_rep_localized = LLM_rep[:, :n_layer][:, mask.astype(bool)]

path_localized = os.path.join(RESULTS_DIR, f'LLM_rep_localized_attention.npy')
np.save(path_localized, LLM_rep_localized)
path_mask = os.path.join(RESULTS_DIR, f'LLM_rep_localized_attention_mask.npy')
np.save(path_mask, mask)


# setting 3: only hidden layers
mask = is_topk(t_values_matrix[n_layer:], k=num_units)
LLM_rep_localized = LLM_rep[:, n_layer:][:, mask.astype(bool)]

path_localized = os.path.join(RESULTS_DIR, f'LLM_rep_localized_hidden.npy')
np.save(path_localized, LLM_rep_localized)
path_mask = os.path.join(RESULTS_DIR, f'LLM_rep_localized_hidden_mask.npy')
np.save(path_mask, mask)