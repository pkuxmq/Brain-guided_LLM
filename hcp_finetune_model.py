import os
import pandas as pd
import pickle
import numpy as np
import torch
import random
from LM_finetune import ModelwithAttentionSupervision
from hcp_utils import *
from torch.cuda.amp import autocast, GradScaler
import argparse
import logging
from collections import defaultdict
import json


parser = argparse.ArgumentParser(description='Finetune model for HCP Relational Task')
# Paths
parser.add_argument('-RESULTS_DIR', type=str, default='./hcp_results/finetune_results/')
parser.add_argument('-LLM_PREV_RESULTS_DIR', type=str, default='./hcp_results/activations_results/')
parser.add_argument('-FMRI_RESULTS_DIR', type=str, default='../hcp_fmri_results/stimdurMeanRT/')

# Validation/Test Sets (New Data)
parser.add_argument('-VAL_DATA', type=str, default='./data/relational_val_set.jsonl')
parser.add_argument('-TEST_DATA', type=str, default='./data/relational_test_set.jsonl')

# Intervention Info for filtering
parser.add_argument('-use_intervention_filter', action='store_true', help='Only use successfully intervened stimuli for guidance')
parser.add_argument('-INTERVENTION_RESULTS_DIR', type=str, default='./hcp_results/intervention_results/')

# Model Config
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/')
parser.add_argument('-lora', action='store_true')
parser.add_argument('-bf16', action='store_true')
parser.add_argument('-device', type=str, default='cuda')
parser.add_argument('-w_device', type=str, default=None)

# Training Hyperparameters
parser.add_argument('-max_epoch', type=int, default=100)
parser.add_argument('-lr', type=float, default=1e-6)
parser.add_argument('-weight_decay', type=float, default=0.)
parser.add_argument('-train_weight', type=float, default=0.1)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-test_batch_size', type=int, default=1)
parser.add_argument('-grad_accumulate_step', type=int, default=1)
parser.add_argument('-scheduler', action='store_true')
parser.add_argument('-forward_partial', action='store_true')

# Loss Weights
parser.add_argument('-use_label', action='store_true', help='Use Label Supervision (CrossEntropy)')
parser.add_argument('-use_fmri', action='store_true', help='Use fMRI Guidance (Regularization)')
parser.add_argument('-label_weight', type=float, default=1.0)
parser.add_argument('-loss_type', type=str, default='cosine', help='cosine or mse for fMRI loss')

# fMRI Guidance Settings
parser.add_argument('-ridge_alpha', type=float, default=100.)
parser.add_argument('-use_ridgecv', action='store_true')
parser.add_argument('-all_layer', action='store_true')
parser.add_argument('-add_intercept', action='store_true')
parser.add_argument('-fit_consistent', action='store_true', help='Only use consistent trials (Model==Human) to train W')
parser.add_argument('-guidance_filter_correct', action='store_true', default=True, help='Only use Human Correct trials for guidance')
parser.add_argument('-use_random_fmri', action='store_true')
parser.add_argument('-use_label_as_fmri', action='store_true')
parser.add_argument('-use_all_stimuli', action='store_true', help='Use all stimuli found in LLM behavior data, even if no fMRI data')
parser.add_argument('-filter_subject', action='store_true')

# Validation Settings
parser.add_argument('-val_epoch', type=int, default=1)
parser.add_argument('-val_all_order', action='store_true', help='Shuffle description order in prompt for Validation/Test')
parser.add_argument('-early_stop_tolerance', type=int, default=20)
parser.add_argument('-pre_eval', action='store_true', help='Run validation/test once before training')
parser.add_argument('-log_stdout', action='store_true', help='Also print logs to stdout')
parser.add_argument('-delete_checkpoint', action='store_true')
parser.add_argument('-resume_checkpoint_path', type=str, default=None)

# Suffix
parser.add_argument('-suffix', type=str, default='')
parser.add_argument('-seed', type=int, default=0)

args = parser.parse_args()

_seed_ = args.seed
random.seed(_seed_)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(_seed_)

# Construct Suffix
suffix = 'nointvinfo'
if args.suffix != '':
    suffix += '_' + args.suffix
if args.use_label: suffix += '_label'
if args.use_fmri: suffix += '_fmri'
suffix += f'_lr{args.lr}'
if args.train_weight != 1.: suffix += f'-train{args.train_weight}'
if args.lora: suffix += '_lora'
if args.bf16: suffix += '_bf16'
if args.fit_consistent: suffix += '_fitconsistent'
suffix += f'_{args.loss_type}'
if args.add_intercept: suffix += '_addintercept'
if args.use_random_fmri: suffix += '_randfmri'
if args.use_label_as_fmri: suffix += '_labelasfmri'
if args.use_all_stimuli: suffix += '_allstim'
if args.val_all_order: suffix += '_valallorder'
if args.filter_subject: suffix += '_filtersubject'
if args.use_intervention_filter: suffix += '_intvfilter'

filtered_subject = ['872158', '200008', '611938', '308129', '984472', '727553', '529549', '771354', '421226', '441939']

RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type, suffix)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

log_path = os.path.join(RESULTS_DIR, 'train.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if args.log_stdout:
    logging.getLogger().addHandler(logging.StreamHandler())
logging.info(args)

device = args.device
w_device = args.w_device if args.w_device is not None else args.device
SHAPE_MAP = {1: "circle", 2: "cross", 3: "triangle", 4: "square", 5: "star", 6: "hexagon"}
TEXTURE_MAP = {1: "ring_dots", 2: "angular_ticks", 3: "zigzag_dashes", 4: "solid_blocks", 5: "knot_weave", 6: "swirl_hooks"}

import itertools

def construct_relational_prompt_local(item_info, mode='default'):
    """
    mode: 
      'default' -> standard order
      'shuffle' -> random shuffle of lines
      'all_permutations' -> return list of messages for all 24 permutations of lines
    """
    # Detect Attributes
    if 'active_attributes' in item_info:
        attr_list = item_info['active_attributes']
        attr1 = attr_list[0]
        attr2 = attr_list[1]
    else:
        # Default Logic
        attr1 = 'shape'
        attr2 = 'texture'

    # Helper
    pos = item_info['positions']
    def get_desc(loc):
        # Generic Retrieval
        p_dict = pos[loc]
        
        val1 = None
        val2 = None
        
        # Try direct attribute name match (New Format)
        if attr1 in p_dict:
             val1 = p_dict[attr1]
        
        if attr2 in p_dict:
             val2 = p_dict[attr2]
             
        # Fallback to Legacy ID (Old Format)
        if val1 is None:
             if attr1 == 'shape': val1 = p_dict.get('shape_id')
             elif attr1 == 'texture': val1 = p_dict.get('texture_id')
             
        if val2 is None:
             if attr2 == 'shape': val2 = p_dict.get('shape_id')
             elif attr2 == 'texture': val2 = p_dict.get('texture_id')

        # Formulate Description Strings
        # Legacy ID Map check
        if attr1 == 'shape' and isinstance(val1, int):
             val1_txt = SHAPE_MAP.get(val1, f"Shape{val1}")
        else:
             val1_txt = str(val1)

        if attr2 == 'texture' and isinstance(val2, int):
             val2_txt = TEXTURE_MAP.get(val2, f"Texture{val2}")
        else:
             val2_txt = str(val2)
            
        return f"{attr1} is {val1_txt}; {attr2} is {val2_txt}"
    
    if mode != 'default':
        shuffle_description = 'The four image descriptions may appear in any order. '
    else:
        shuffle_description = ''

    prompt_relational = (
        "You are an expert in relational reasoning. "
        f"I will provide descriptions of four images (A, B, C, D), which describe two attributes of each image: "
        f"{attr1} and {attr2}. {shuffle_description}"
        "Image A and Image B differ in exactly one attribute "
        f"(either {attr1} or {attr2}). "
        "Your task is to determine whether Image C and Image D also differ in that attribute. "
        f"For example, if Image A and Image B have different {attr1}s, then you should determine if Image C and Image D also have different {attr1}s. "
        f"Or if Image A and Image B have different {attr2}s, then you should determine if Image C and Image D also have different {attr2}s. "
        "If the statement is true, you should answer 'True'; otherwise answer 'False'. "
        "Only answer 'True' or 'False' at the beginning of the response. "
    )
    
    lines = [
        f"Image A: {get_desc('TopLeft')}.",
        f"Image B: {get_desc('TopRight')}.",
        f"Image C: {get_desc('BottomLeft')}.",
        f"Image D: {get_desc('BottomRight')}."
    ]
    
    question_text = f"For the attribute that Image A and Image B differ in (either {attr1} or {attr2}), do Image C and Image D also differ in that attribute? Only answer 'True' or 'False' at the beginning of your response."

    def build_msgs(permuted_lines):
        stimulus_desc = ""
        for i, line in enumerate(permuted_lines):
            stimulus_desc += f"{i+1}. {line}\n"
        stimulus_desc += question_text
        return [
            {"role": "user", "content": prompt_relational},
            {"role": "assistant", "content": "Understood. I will analyze the attributes and perform relational reasoning to judge the statement. I will only answer 'True' or 'False' at the beginning of the response."},
            {"role": "user", "content": stimulus_desc},
        ]

    if mode == 'all_permutations':
        all_perms = list(itertools.permutations(lines))
        return [build_msgs(p) for p in all_perms]
    
    if mode == 'shuffle':
        random.shuffle(lines)
    
    return build_msgs(lines)

# 1. Model Configuration
# ----------------------
max_new_tokens = 1
parse_model_type = 'default'
model_path = args.model_path
if args.model_type == 'llama2':
    parse_model_type = 'llama2'
    layer_num = 32
    max_new_tokens = 2
elif args.model_type == 'llama3':
    layer_num = 32
elif args.model_type == 'mistral':
    layer_num = 32
elif args.model_type == 'qwen1-5b':
    layer_num = 28
elif args.model_type == 'qwen7b':
    layer_num = 28
elif args.model_type == 'qwen72b':
    layer_num = 80
elif args.model_type == 'llama3-3_70b':
    layer_num = 80
elif args.model_type == 'qwen3_4b':
    parse_model_type = 'qwen3'
    layer_num = 36
elif args.model_type == 'phi4-mini':
    layer_num = 32
elif args.model_type == 'gemma2_9b':
    layer_num = 42
else:
    raise("Error! Unsupported model type")

if args.all_layer:
    layer_index_all = list(range(0, layer_num))
else:
    layer_index_all = list(range(layer_num//4, layer_num*3//4))

logging.info(f'Layer index for training: {layer_index_all}')

# Initialize Model
model = ModelwithAttentionSupervision(args.model_path, layer_index_all, device, False, args.lora, args.bf16, args.model_type)

# Optimizer
optimizer_parameters = [p for p in model.model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr, weight_decay=args.weight_decay)

# Resume
if args.resume_checkpoint_path:
    model.model.load_state_dict(torch.load(os.path.join(args.resume_checkpoint_path, 'model-max.pth'), map_location='cpu'))
    optimizer.load_state_dict(torch.load(os.path.join(args.resume_checkpoint_path, 'optimizer-max.pth'), map_location='cpu'))

# 2. Data Loading & Alignment (fMRI Guidance Preparation)
# --------------------------------------------------------
# Prepare fMRI Guidance Dictionary
# Dictionary Structure: stim_name -> layer -> List of supervision dicts {'W', 'b', 'fmri_state', 'loss_type'}
guidance_data_full = defaultdict(lambda: defaultdict(list)) 
training_stimuli_info = {} # Key: Stimulus Name -> Info Dict

# Load Intervention Info if needed
intervention_success_dict = {}
if args.use_intervention_filter:
    logging.info("Loading Intervention Info for Filtering...")
    # Determine Path
    idx_path = os.path.join(args.INTERVENTION_RESULTS_DIR, 'intervention_index_info.pkl')

    if os.path.exists(idx_path):
        with open(idx_path, 'rb') as f:
            # Structure: {'Relational': {sub: {'indices': [...], 'stimuli': [...]}}} (New)
            # Or {'Relational': {sub: [indices]}} (Old)
            intervention_success_data = pickle.load(f)
            intervention_success_dict = intervention_success_data.get('Relational', {})
        logging.info(f"Loaded intervention success data for {len(intervention_success_dict)} subjects.")
    else:
        raise FileNotFoundError(f"Could not find intervention_index_info.pkl at {idx_path}. Please provide the specific intervention result directory.")

if args.use_all_stimuli:
    logging.info("Loading All Component Info (use_all_stimuli)...")
    LLM_PREV_RES_PATH = os.path.join(args.LLM_PREV_RESULTS_DIR, args.model_type)
    try:
        _, llm_behav = load_relational_llm_data(LLM_PREV_RES_PATH)
        for k, v in llm_behav.items():
            training_stimuli_info[k] = v['stimulus_info']
        logging.info(f"Loaded {len(training_stimuli_info)} stimuli from LLM behavior records.")
    except Exception as e:
        logging.warning(f"Could not load LLM behavior data: {e}. Training limited to available sources.")

if args.use_fmri:
    logging.info("Preparing fMRI Guidance...")
    
    # Load Global Activations
    LLM_PREV_RESULTS_DIR = os.path.join(args.LLM_PREV_RESULTS_DIR, args.model_type)
    llm_acts, llm_behav = load_relational_llm_data(LLM_PREV_RESULTS_DIR)
    
    # Get Subjects
    target_subs = filtered_subject if args.filter_subject else None
    subjects = get_relational_fmri_subjects(args.FMRI_RESULTS_DIR, target_subjects=target_subs)
    logging.info(f"Found {len(subjects)} subjects.")
    
    num_fmri_guidance = 0
    
    for sub_info in subjects:
        sub_id = sub_info['subject']
        # Align
        data = align_subject_data(sub_info, llm_acts, llm_behav)
        if data is None: continue
        
        logging.info(f"Processing Subject {sub_id}...")
        
        # Get Intervention Success Set for this subject
        success_stimuli_set = set()
        if args.use_intervention_filter:
            if sub_id in intervention_success_dict:
                sub_data = intervention_success_dict[sub_id]
                if isinstance(sub_data, dict):
                    if 'stimuli' in sub_data:
                        success_stimuli_set = set(sub_data['stimuli'])
                    elif 'indices' in sub_data:
                        # Fallback to indices if stimuli not present but dict wrap exists
                        for idx in sub_data['indices']:
                            if idx < len(data['stimuli']):
                                success_stimuli_set.add(data['stimuli'][idx])
                elif isinstance(sub_data, list):
                    # Old Format (Indices list)
                    for idx in sub_data:
                        if idx < len(data['stimuli']):
                            success_stimuli_set.add(data['stimuli'][idx])
            
            logging.info(f"  Subject {sub_id}: Found {len(success_stimuli_set)} successful intervention stimuli.")
            
        # Unpack
        X = data['llm_rep'] # N x L x D
        Y = data['fmri_data'] # N x V
        
        if args.use_random_fmri:
            Y = np.random.randn(*Y.shape)
            
        if args.use_label_as_fmri:
            # Synthetic 2D target
            Y_syn = np.zeros((Y.shape[0], 2))
            for i in range(Y.shape[0]):
                # data['stimulus_info'][i] has label/answer
                inf = data['stimulus_info'][i]
                # Determine ground truth
                is_true = False
                l = inf.get('label')
                if l == 1:
                    is_true = True
                elif l == 0:
                    is_true = False
                else:
                    if 'YES' in data['stimuli'][i].upper():
                        is_true = True
                
                if is_true: 
                    Y_syn[i, 1] = 1.0 # True class
                else:
                    Y_syn[i, 0] = 1.0 # False class
            Y = Y_syn
        
        model_acc = data['model_acc']
        human_acc = data['human_acc']
        stimuli = data['stimuli']
        infos = data['stimulus_info'] # List of dicts
        
        # Determine Training Indices for W
        # If fit_consistent, use trials where Model == Human (Correct or Wrong)
        if args.fit_consistent:
            idx_train = np.where(model_acc == human_acc)[0]
        else:
            idx_train = np.arange(len(model_acc))
            
        if len(idx_train) < 10:
            logging.warning(f"Not enough trials to train W for {sub_id}. Skipping.")
            continue
            
        # Train W per layer
        for layer in layer_index_all:
            if args.use_ridgecv:
                if args.add_intercept:
                    W, b = get_W_info_cv(X, Y, layer, idx_train, return_b=True)
                else:
                    W = get_W_info_cv(X, Y, layer, idx_train, return_b=False)
                    b = 0
            else:
                if args.add_intercept:
                    W, b = get_W_info(X, Y, layer, idx_train, ridge_alpha=args.ridge_alpha, return_b=True)
                else:
                    W = get_W_info(X, Y, layer, idx_train, ridge_alpha=args.ridge_alpha, return_b=False)
                    b = 0
            
            # Convert to Tensor (Keeping on CPU for storage to save VRAM, move to GPU in loop)
            # Use float32 to match LM_finetune expectations usually
            if args.bf16:
                W_t = torch.from_numpy(W).to(torch.bfloat16).to(w_device)
            else:
                W_t = torch.from_numpy(W).float().to(w_device)
            
            if isinstance(b, (np.ndarray, float, int)):
                if args.bf16:
                    b_t = torch.tensor(b, dtype=torch.bfloat16).to(w_device)
                else:
                    b_t = torch.tensor(b, dtype=torch.float32).to(w_device)
            else:
                b_t = torch.tensor(0., dtype=torch.bfloat16 if args.bf16 else torch.float32).to(w_device)
                
            # Store Guidance for "Human Correct" trials
            if args.guidance_filter_correct:
                idx_guide = np.where((human_acc == 1) & (model_acc != 1))[0]
            else:
                idx_guide = np.arange(len(human_acc))
                
            for i in idx_guide:
                stim_name = stimuli[i]
                
                # Check Intervention Criteria
                if args.use_intervention_filter:
                    if stim_name not in success_stimuli_set:
                        continue # Skip this trial because intervention failed or not attempted

                target_fmri = Y[i]
                if args.bf16:
                    target_fmri_t = torch.from_numpy(target_fmri).to(torch.bfloat16).to(w_device)
                else:
                    target_fmri_t = torch.from_numpy(target_fmri).float().to(w_device)
                
                # Append to list for this layer
                guidance_data_full[stim_name][layer].append({
                    'W': W_t,
                    'b': b_t,
                    'fmri_state': target_fmri_t,
                    'loss_type': args.loss_type
                })

        num_fmri_guidance += len(idx_guide)
                
        # Ensure ALL stimuli (including Model Correct ones) are registered for Label Supervision
        for i in range(len(stimuli)):
            stim_name = stimuli[i]
            if stim_name not in training_stimuli_info:
                training_stimuli_info[stim_name] = infos[i]

    logging.info(f"Guidance prepared for {len(training_stimuli_info)} stimuli.")
    logging.info(f"Number of fMRI guidance trials: {num_fmri_guidance}")

# 3. Construct Training Dataset
# ------------------------------
signal_dict = [] # List of {'input_ids', 'label_id', 'stim_name', 'y'}

# Helper to load all keys if not using fMRI to guide discovery
if not args.use_fmri and args.use_label:
    # Fallback
    if not training_stimuli_info:
        LLM_PREV_RESULTS_DIR = os.path.join(args.LLM_PREV_RESULTS_DIR, args.model_type)
        llm_acts, llm_behav = load_relational_llm_data(LLM_PREV_RESULTS_DIR)
        
        # If align with fMRI (even without use_fmri), filter out stimuli not in subjects
        # This requires loading subjects and simulating alignment mapping
        if not args.use_all_stimuli:
            logging.info("Aligning Pure-Label Training Set Maximum to Subject Data...")
            target_subs = filtered_subject if args.filter_subject else None
            subjects = get_relational_fmri_subjects(args.FMRI_RESULTS_DIR, target_subjects=target_subs)
            valid_stimuli = set()
            for sub_info in subjects:
                # Mock Alignment to get valid stimuli list
                data = align_subject_data(sub_info, llm_acts, llm_behav)
                if data is not None:
                    valid_stimuli.update(data['stimuli'])
            
            # Filter
            for k in sorted(list(valid_stimuli)):
                 if k in llm_behav:
                      training_stimuli_info[k] = llm_behav[k]['stimulus_info']
            logging.info(f"Filtered Pure Label Dataset to {len(training_stimuli_info)} stimuli (Subject Intersection).")
            
        else:
             for k, v in sorted(llm_behav.items()):
                training_stimuli_info[k] = v['stimulus_info']

logging.info("Building Training Set...")
for stim_name, info in training_stimuli_info.items():
    # Construct Prompt
    try:
        messages = construct_relational_prompt_local(info, mode='default')
        
        # Tokenize Input
        if args.model_type == 'llama2':
            text = model.tokenizer.apply_chat_template(messages, tokenize=False)
            text += ' '
            encodeds = model.tokenizer([text], return_tensors='pt')
            text_inputs = encodeds['input_ids']
        elif 'qwen3' in args.model_type:
             text_inputs = model.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=True, enable_thinking=False)
        else:
             text_inputs = model.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=True)
             
        # Create Label (Next Token)
        # Identify Gound Truth
        label_str = None
        # Try metadata label
        l = info.get('label')
        if l == 1: label_str = 'True'
        elif l == 0: label_str = 'False'
        
        if label_str is None:
             if 'YES' in stim_name.upper(): label_str = 'True'
             elif 'NO' in stim_name.upper(): label_str = 'False'
             
        if label_str is None: continue # Skip if no label
        
        # Tokenize Label
        label_token = model.tokenizer(label_str, return_tensors='pt')['input_ids']
        if label_token.shape[-1] > 1:
            label_id = label_token[0, -1] 
        else:
            label_id = label_token[0, 0]
            
        # supervision dict ('y')
        y_info = guidance_data_full[stim_name] # Dict[layer] -> list of sup
        
        # If use_fmri is True, but y_info is empty (no guidance), and use_label is False. 
        # Then this item contributes NOTHING. Skip.
        if args.use_fmri and not args.use_label:
             if not y_info:
                 continue

        signal_dict.append({
            'input_ids': text_inputs[0], # 1D tensor
            'label_id': label_id,
            'stim_name': stim_name,
            'y': y_info # Passed to attention_supervision_dict
        })
        
    except Exception as e:
        logging.warning(f"Error preparing item {stim_name}: {e}")

logging.info(f"Training set size: {len(signal_dict)}")

# 4. Validation Helper
# --------------------
def validate_batch(epoch):
    logging.info("Validating...")
    model.model.eval()
    
    val_items = []
    if args.VAL_DATA.endswith('.jsonl'):
        with open(args.VAL_DATA, 'r') as f:
            for line in f:
                if line.strip(): val_items.append(json.loads(line))
    else:
        with open(args.VAL_DATA, 'r') as f:
            val_items = json.load(f)
            
    # Prepare item -> permutation messages
    item_meta = {}
    item_scores = defaultdict(list)
    
    batch_msgs = []
    batch_meta = []
    
    for item_idx, item in enumerate(val_items):
        try:
            gt = -1
            l_raw = item.get('label')
            if isinstance(l_raw, int):
                gt = l_raw
            elif isinstance(l_raw, str):
                if l_raw.lower() in ['true', 'yes', '1']: gt = 1
                elif l_raw.lower() in ['false', 'no', '0']: gt = 0
            if gt == -1:
                continue
            
            item_meta[item_idx] = {'gt': gt}
            
            if args.val_all_order:
                msgs_list = construct_relational_prompt_local(item, mode='all_permutations')
            else:
                msgs_list = [construct_relational_prompt_local(item, mode='default')]
            
            for msg in msgs_list:
                batch_msgs.append(msg)
                batch_meta.append(item_idx)
                
                if len(batch_msgs) >= args.test_batch_size:
                    with torch.no_grad():
                        batch_ans = model.generate_ans_batch(batch_msgs, max_new_tokens=max_new_tokens, parse_model_type=parse_model_type)
                    
                    for ans, itm_idx in zip(batch_ans, batch_meta):
                        val_pred = parse_relational_response(ans, args.model_type)
                        gt_val = item_meta[itm_idx]['gt']
                        item_scores[itm_idx].append(1 if val_pred == gt_val else 0)
                    
                    batch_msgs = []
                    batch_meta = []
        except Exception as e:
            logging.warning(f"Val Error: {e}")
            continue
    
    if batch_msgs:
        with torch.no_grad():
            batch_ans = model.generate_ans_batch(batch_msgs, max_new_tokens=max_new_tokens, parse_model_type=parse_model_type)
        for ans, itm_idx in zip(batch_ans, batch_meta):
            val_pred = parse_relational_response(ans, args.model_type)
            gt_val = item_meta[itm_idx]['gt']
            item_scores[itm_idx].append(1 if val_pred == gt_val else 0)

    # Aggregate
    correct = 0
    total = 0
    for itm_idx, scores in item_scores.items():
        if len(scores) == 0:
            continue
        avg_acc = np.mean(scores)
        correct += avg_acc
        total += 1

    acc = correct / total if total > 0 else 0
    logging.info(f"Epoch {epoch} Validation Acc: {acc:.4f} ({correct:.2f}/{total})")
    model.model.train()
    return acc

def load_relational_items(path):
    items = []
    if path.endswith('.jsonl'):
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    else:
        with open(path, 'r') as f:
            items = json.load(f)
    return items

def eval_relational_test(items, log_prefix="Test"):
    model.model.eval()

    # Batch Test
    correct_by_cat = defaultdict(list)
    all_corect = []

    # Prepare batches
    batch_msgs_permutations = []
    for item in items:
        gt = -1
        l_raw = item.get('label')
        if isinstance(l_raw, int): gt = l_raw
        elif isinstance(l_raw, str):
            if l_raw.lower() in ['true', 'yes', '1']: gt = 1
            elif l_raw.lower() in ['false', 'no', '0']: gt = 0
        if gt == -1: continue

        cat = item.get('category', 'unknown')
        if args.val_all_order:
            msgs_list = construct_relational_prompt_local(item, mode='all_permutations')
        else:
            msgs_list = [construct_relational_prompt_local(item, mode='default')]
        #msgs_list = construct_relational_prompt_local(item, mode='all_permutations')
        batch_msgs_permutations.append({
            'gt': gt,
            'cat': cat,
            'msgs': msgs_list
        })

    current_batch_msgs = []
    current_batch_meta = []

    item_scores = defaultdict(list)
    item_meta = {}

    for i, item_obj in enumerate(batch_msgs_permutations):
        item_meta[i] = {'gt': item_obj['gt'], 'cat': item_obj['cat']}
        for p_i, msg in enumerate(item_obj['msgs']):
            current_batch_msgs.append(msg)
            current_batch_meta.append((i, p_i))

            if len(current_batch_msgs) >= args.test_batch_size:
                with torch.no_grad():
                    batch_ans = model.generate_ans_batch(current_batch_msgs, max_new_tokens=max_new_tokens, parse_model_type=parse_model_type)

                for ans, (itm_idx, _) in zip(batch_ans, current_batch_meta):
                    val_pred = parse_relational_response(ans, args.model_type)
                    gt = item_meta[itm_idx]['gt']
                    item_scores[itm_idx].append(1 if val_pred == gt else 0)

                current_batch_msgs = []
                current_batch_meta = []

    if current_batch_msgs:
        with torch.no_grad():
            batch_ans = model.generate_ans_batch(current_batch_msgs, max_new_tokens=max_new_tokens, parse_model_type=parse_model_type)
        for ans, (itm_idx, _) in zip(batch_ans, current_batch_meta):
            val_pred = parse_relational_response(ans, args.model_type)
            gt = item_meta[itm_idx]['gt']
            item_scores[itm_idx].append(1 if val_pred == gt else 0)

    for itm_idx, scores in item_scores.items():
        avg_acc = np.mean(scores)
        all_corect.append(avg_acc)
        cat = item_meta[itm_idx]['cat']
        correct_by_cat[cat].append(avg_acc)

    final_acc = np.mean(all_corect) if all_corect else 0
    logging.info(f"{log_prefix} Accuracy: {final_acc:.4f}")
    model.model.train()
    return final_acc, correct_by_cat

# 5. Training Loop
# ----------------
if args.scheduler:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

best_val_acc = 0.0
best_val_epoch = -1
patience_counter = 0
prev_val_acc = 0.0 # Track previous epoch accuracy for relative early stopping

logging.info("Starting Training...")
model.model.train()

if args.pre_eval:
    logging.info("Running Pre-Train Validation...")
    _ = validate_batch(epoch=-1)
    logging.info("Running Pre-Train Test...")
    test_items = load_relational_items(args.TEST_DATA)
    _ = eval_relational_test(test_items, log_prefix="Pre-Train Test")

for epoch in range(args.max_epoch):
    random.shuffle(signal_dict)
    
    epoch_loss_total = 0
    epoch_loss_label = 0
    epoch_fmri_loss = 0
    steps = 0
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(signal_dict):
        input_ids = batch['input_ids'].unsqueeze(0).to(device) # 1 x Seq
        label_id = batch['label_id'].unsqueeze(0).to(device) # 1
        stim_name = batch['stim_name']
        y_guidance = batch['y'] # attention_supervision_dict
        if args.use_fmri:
            # Filter out empty supervision lists to avoid index errors in LM_finetune
            y_guidance = {k: v for k, v in y_guidance.items() if isinstance(v, list) and len(v) > 0}
        
        # Two forward passes structure as per finetune_model_sep
        # Pass 1: Regularization (fMRI) - if active
        
        l_fmri = 0
        loss = 0
        
        if args.use_fmri and y_guidance:
            attention_mask = None # Provide if needed
            
            # Use forward_partial if requested/available or standard forward
            if args.forward_partial:
                 if args.bf16:
                     with autocast(dtype=torch.bfloat16):
                        l = model.forward_partial(input_ids, attention_mask, attention_supervision_dict=y_guidance)
                 else:
                     l = model.forward_partial(input_ids, attention_mask, attention_supervision_dict=y_guidance)
            else:
                 if args.bf16:
                     with autocast(dtype=torch.bfloat16):
                        l = model(input_ids, attention_mask, attention_supervision_dict=y_guidance)
                 else:
                     l = model(input_ids, attention_mask, attention_supervision_dict=y_guidance)
            
            if l is not None:
                # Apply weight implies adding it to loss
                # Here we treat reg as main loss component if use_fmri
                l = l * args.train_weight / args.grad_accumulate_step
                loss += l
                l_fmri = l.item()
                epoch_fmri_loss += l_fmri
        
        if loss != 0:
             loss.backward()
             epoch_loss_total += loss.item()
             
        # Pass 2: Label
        if args.use_label:
            # Condition check (epoch freq etc)
            pass_label = True
            if pass_label:
                if args.bf16:
                     with autocast(dtype=torch.bfloat16):
                        l_lbl = model(input_ids, None, label=label_id) * args.label_weight / args.grad_accumulate_step
                else:
                     l_lbl = model(input_ids, None, label=label_id) * args.label_weight / args.grad_accumulate_step
                
                loss_label_val = l_lbl.item()
                epoch_loss_label += loss_label_val
                l_lbl.backward()

        if (i + 1) % args.grad_accumulate_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
            
    if args.scheduler:
        lr_scheduler.step()
        
    logging.info(f"Epoch {epoch} Loss Total: {epoch_loss_total/len(signal_dict):.4f} (L: {epoch_loss_label/len(signal_dict):.4f}, fMRI: {epoch_fmri_loss/len(signal_dict):.4f})")
    
    if (epoch + 1) % args.val_epoch == 0:
        val_acc = validate_batch(epoch)
        
        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            torch.save(model.model.state_dict(), os.path.join(RESULTS_DIR, 'model-max.pth'))
            torch.save(optimizer.state_dict(), os.path.join(RESULTS_DIR, 'optimizer-max.pth'))
            
        # Early Stopping Logic (Based on finetune_model_sep.py: compare to PREVIOUS val_acc)
        # If accuracy decreases compared to last check, increment counter
        if val_acc < prev_val_acc:
            patience_counter += 1
        elif val_acc > prev_val_acc:
            patience_counter = 0
            
        prev_val_acc = val_acc
        
        if patience_counter >= args.early_stop_tolerance:
             logging.info("Early Stopping.")
             break
                 
logging.info("Done.")

# 6. Final Test
# -------------
logging.info("Running Final Test")
# Log best checkpoint epoch info (if available)
if best_val_epoch >= 0:
    logging.info(f"Loading best checkpoint from epoch {best_val_epoch}")
# Load Best Model
model.model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'model-max.pth'), map_location='cpu'))

# Save model (align with previous pipeline)
save_model_path = os.path.join(RESULTS_DIR, 'model')
if args.lora:
    model.merge_lora()
model.model.save_pretrained(save_model_path)
model.tokenizer.save_pretrained(save_model_path)

if args.delete_checkpoint:
    if os.path.exists(os.path.join(RESULTS_DIR, 'model-max.pth')):
        os.remove(os.path.join(RESULTS_DIR, 'model-max.pth'))
    if os.path.exists(os.path.join(RESULTS_DIR, 'optimizer-max.pth')):
        os.remove(os.path.join(RESULTS_DIR, 'optimizer-max.pth'))
model.model.eval()

test_items = load_relational_items(args.TEST_DATA)
final_acc, correct_by_cat = eval_relational_test(test_items, log_prefix="Final Test")

for cat, scores in correct_by_cat.items():
    logging.info(f"Category {cat}: {np.mean(scores):.4f} ({len(scores)})")

path_results = os.path.join(RESULTS_DIR, f'test_behaviour_results.pkl')
with open(path_results, "wb") as f:
    pickle.dump(correct_by_cat, f)



