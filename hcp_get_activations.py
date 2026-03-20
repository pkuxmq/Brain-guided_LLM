import os
import pandas as pd
import pickle
import json
from LM import LM_nnsight, LM_untrained_nnsight
import numpy as np
import argparse

# Mapping from ID (int) to Description (str)
SHAPE_MAP = {
    1: "circle",
    2: "cross",
    3: "triangle",
    4: "square",
    5: "star",
    6: "hexagon"
}

TEXTURE_MAP = {
    1: "ring_dots",
    2: "angular_ticks",
    3: "zigzag_dashes",
    4: "solid_blocks",
    5: "knot_weave",
    6: "swirl_hooks"
}
# ==========================================

parser = argparse.ArgumentParser(description='Get model activations')
parser.add_argument('-RESULTS_DIR', type=str, default='./hcp_results/activations_results/') # path to save results
parser.add_argument('-model_type', type=str, default='mistral')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Mistral-7B-Instruct-v0.2/')
parser.add_argument('-untrained', action='store_true')
parser.add_argument('-device', type=str, default='cpu')
parser.add_argument('-llama2_type', action='store_true')
parser.add_argument('-bf16', action='store_true')

args = parser.parse_args()

if args.untrained:
    LM_model = LM_untrained_nnsight
    RESULTS_DIR = os.path.join(args.RESULTS_DIR, 'untrained', args.model_type)
else:
    LM_model = LM_nnsight
    RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

model_path = args.model_path
device = args.device

max_new_tokens = 1
if args.model_type in ['llama2', 'llama2-70b']:
    if args.llama2_type:
        model = LM_model(model_path, device, parse_model_type='llama2', bf16=args.bf16)
    else:
        model = LM_model(model_path, device, bf16=args.bf16)
    # llama2 will first output ' ' before the answer, so it requires 2 tokens
    max_new_tokens = 2
elif 'deepseek' in args.model_type:
    model = LM_model(model_path, device, parse_model_type='deepseek', bf16=args.bf16)
    max_new_tokens = 1000
elif 'qwen3' in args.model_type:
    model = LM_model(model_path, device, parse_model_type='qwen3', bf16=args.bf16)
else:
    model = LM_model(model_path, device, bf16=args.bf16)

# Load stimuli mapping
stimuli_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stimuli_mapping_final.jsonl'))

print(f"Loading stimuli from {stimuli_path}")

filtered_stimuli = []
with open(stimuli_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        if item['stimulus_file'].startswith('int_') and item['stimulus_file'].endswith('.bmp'):
            filtered_stimuli.append(item)

# Prompt template
prompt_relational = (
    "You are an expert in relational reasoning. "
    "I will provide descriptions of four images (A, B, C, D), which describe two attributes of each image: shape and texture. "
    "Image A and Image B differ in exactly one attribute (either shape or texture). "
    "Your task is to determine whether Image C and Image D also differ in that attribute. "
    "For example, if Image A and Image B have different shapes, then you should determine if Image C and Image D also have different shapes. "
    "Or if Image A and Image B have different textures, then you should determine if Image C and Image D also have different textures. "
    "If the statement is true, you should answer 'True'; otherwise answer 'False'. "
    "Only answer 'True' or 'False' at the beginning of the response. "
)

# Output dictionaries
# Key: stimulus_file (str) -> Value: Result
results_activations = {} 
results_behavior = {}

print('Getting activations for ' + args.model_type)
print(f'Found {len(filtered_stimuli)} relational stimuli.')
print('\n')

correct_count = 0
total_count = 0

for item in filtered_stimuli:
    stim_name = item['stimulus_file']
    print(f"Processing {stim_name}...")
    
    # Extract Ground Truth Label from filename
    # filenames usually contain YES or NO (e.g., int_..._YES_....bmp)
    ground_truth = None
    if 'YES' in stim_name.upper():
        ground_truth = 1
    elif 'NO' in stim_name.upper():
        ground_truth = 0
    else:
        print(f"Warning: Could not extract ground truth YES/NO from filename {stim_name}")
    
    pos = item['positions']
    
    # Helper to clean text
    def get_desc(loc):
        s_id = pos[loc]['shape_id']
        t_id = pos[loc]['texture_id']
        s_txt = SHAPE_MAP.get(s_id, f"Shape{s_id}")
        t_txt = TEXTURE_MAP.get(t_id, f"Texture{t_id}")
        return f"shape is {s_txt}; texture is {t_txt}"

    # Construct stimulus description
    stimulus_desc = (
        #f"The attributes of four images are:\n"
        f"1. Image A: {get_desc('TopLeft')}.\n"
        f"2. Image B: {get_desc('TopRight')}.\n"
        f"3. Image C: {get_desc('BottomLeft')}.\n"
        f"4. Image D: {get_desc('BottomRight')}.\n"
        f"For the attribute that Image A and Image B differ in (either shape or texture), do Image C and Image D also differ in that attribute? Only answer 'True' or 'False' at the beginning of your response."
    )

    messages = [
        {"role": "user", "content": prompt_relational},
        {"role": "assistant", "content": "Understood. I will analyze the attributes and perform relational reasoning to judge the statement. I will only answer 'True' or 'False' at the beginning of the response."},
        {"role": "user", "content": stimulus_desc},
    ]

    ans, all_tokens = model(messages, max_new_tokens=max_new_tokens, get_all_tokens=True)
    print(ans)
    
    # Answer Parsing
    if args.model_type in ['deepseekqwen1-5b']:
        print(ans)
        ans_ori = ans
        # first parse answer
        idx = ans_ori.rfind('</think>')
        if idx != -1:
            ans = ans_ori[idx+len('</think>'):].strip('\n').strip()
        
        # DeepSeek specific logic for Relational Task (Yes/No)
        # Check for Yes/No keywords if not clean
        idx_y = -1
        idx_n = -1
        if 'true' in ans.lower() or 'yes' in ans.lower():
            idx_y = max(ans.rfind('True'), ans.rfind('true'), ans.rfind('Yes'), ans.rfind('yes'))
        if 'false' in ans.lower() or 'no' in ans.lower():
            idx_n = max(ans.rfind('False'), ans.rfind('false'), ans.rfind('No'), ans.rfind('no'))
            
        if idx_y != -1 and idx_n == -1:
            ans = 'Yes'
        elif idx_n != -1 and idx_y == -1:
            ans = 'No'
        elif idx_n == -1 and idx_y == -1:
            # Fallback for some common phrasing if needed, but Yes/No is standard
            if 'consistent' in ans.lower() and 'not' not in ans.lower():
                ans = 'Yes'
            elif 'inconsistent' in ans.lower() or 'not consistent' in ans.lower():
                ans = 'No'

    model_ans_val = None
    # Normalize checks
    ans_clean = ans.lower().strip().strip("'").strip('"').strip('.')
    if ans_clean in ['yes', 'true', '1']:
        model_ans_val = 1
    elif ans_clean in ['no', 'false', '0']:
        model_ans_val = 0
    else:
        # Fallback check for containment if direct match failed
        if 'yes' in ans_clean or 'true' in ans_clean:
            model_ans_val = 1
        elif 'no' in ans_clean or 'false' in ans_clean:
            model_ans_val = 0
        else:
            print(f"Warning: ambiguous answer for {stim_name}: {ans}")
            model_ans_val = -1 # Indicate error/ambiguity

    # Check correctness
    is_correct = False
    if ground_truth is not None and model_ans_val != -1:
        if model_ans_val == ground_truth:
            is_correct = True
            correct_count += 1
        total_count += 1

    # Extract Representations (match legacy pipeline: attention + hidden)
    if args.model_type in ['deepseekqwen1-5b']:
        # Use answer-token states and average across generated tokens
        state_h, state_a = model.get_all_ans_states_with_tokens(tokens=all_tokens[:-1], prompt=messages)
        state_h = np.mean(state_h, axis=1)
        state_a = np.mean(state_a, axis=1)
    else:
        state_h, state_a = model.get_all_states_with_tokens(tokens=all_tokens[:-1])
        # Take last token (prompt end)
        state_h = state_h[:, -1]
        state_a = state_a[:, -1]

    # Flatten attention heads: (Layers, Heads, DimHead) -> (Layers, Dim)
    state_a = np.reshape(state_a, (state_a.shape[0], -1))

    # Save to memory (store both, concat later to match legacy load logic)
    results_activations[stim_name] = {
        'attention': state_a,
        'hidden': state_h
    }
    
    results_behavior[stim_name] = {
        'model_answer_raw': ans,
        'model_answer_val': model_ans_val,
        'label': ground_truth,
        'correct': is_correct,
        'stimulus_info': item
    }

# Save Results
if total_count > 0:
    print('--------------------')
    print(f'Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.4f}')
    print('--------------------')

print('Saving results...')
# Save as Dictionary pickle
path_activations = os.path.join(RESULTS_DIR, 'hcp_relational_activations.pkl')
path_behavior = os.path.join(RESULTS_DIR, 'hcp_relational_behavior.pkl')

with open(path_activations, "wb") as f:
    pickle.dump(results_activations, f)

with open(path_behavior, "wb") as f:
    pickle.dump(results_behavior, f)

print('Done.')

