import os
import pandas as pd
import pickle
import numpy as np
import torch
import random
from LM_finetune import ModelwithAttentionSupervision
from utils import *
from torch.cuda.amp import autocast, GradScaler
import argparse
import re
import logging

parser = argparse.ArgumentParser(description='Finetune model only with label')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/')
parser.add_argument('-RESULTS_DIR', type=str, default='./results/finetune_results_onlylabel_syntheticdata/')
parser.add_argument('-LLM_PREV_RESULTS_DIR', type=str, default='./results/activations_results/')
parser.add_argument('-model_type', type=str, default='qwen1-5b')
parser.add_argument('-model_path', type=str, default='/data2/huggingface/Qwen2-1.5B-Instruct/')

parser.add_argument('-lora', action='store_true')
parser.add_argument('-bf16', action='store_true')

parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-lr', type=float, default=1e-6)
parser.add_argument('-weight_decay', type=float, default=0.)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-test_batch_size', type=int, default=1)
parser.add_argument('-grad_accumulate_step', type=int, default=1)
parser.add_argument('-scheduler', action='store_true')

parser.add_argument('-val_epoch', type=int, default=10)

parser.add_argument('-resume_model_path', type=str, default=None)
parser.add_argument('-resume_optimizer_path', type=str, default=None)
parser.add_argument('-resume_checkpoint_path', type=str, default=None)

parser.add_argument('-all_layer', action='store_true')

parser.add_argument('-VAL_DATA', type=str, default='./data/deductive_reasoning_data_val_new.json')
parser.add_argument('-TEST_DATA', type=str, default='./data/deductive_reasoning_data_test.json')
parser.add_argument('-synthetic_data_file', type=str, default=None)

parser.add_argument('-balance_label_class', action='store_true')

parser.add_argument('-only_syll', action='store_true')
parser.add_argument('-only_tran', action='store_true')

parser.add_argument('-val_all_order', action='store_true')

parser.add_argument('-suffix', type=str, default='')

parser.add_argument('-device', type=str, default='cpu')

parser.add_argument('-delete_checkpoint', action='store_true')
parser.add_argument('-log_stdout', action='store_true', help='Also print logs to stdout')
parser.add_argument('-seed', type=int, default=0)

args = parser.parse_args()

_seed_ = args.seed
random.seed(_seed_)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(_seed_)

DATA_DIR = args.DATA_DIR
suffix = 'onlylabel'
if args.only_syll:
    suffix = 'onlysyll_' + suffix
if args.only_tran:
    suffix = 'onlytran_' + suffix
if args.lora:
    suffix += '_lora'
if args.bf16:
    suffix += '_bf16'
suffix += f'_e{args.epoch}'
suffix += f'_lr{args.lr}'
if args.weight_decay > 0:
    suffix += f'-wd'
if args.grad_accumulate_step > 1:
    suffix += f'_gradstep-{args.grad_accumulate_step}'
if args.scheduler:
    suffix += f'_scheduler'
if args.balance_label_class:
    suffix += '_balance-class'
if args.val_all_order:
    suffix += '_valallorder'
if args.suffix != '':
    suffix += '_' + args.suffix
RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type, suffix)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

log_path = os.path.join(RESULTS_DIR, 'train.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if args.log_stdout:
    logging.getLogger().addHandler(logging.StreamHandler())
device = args.device

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

logging.info('layer index for training:')
logging.info(layer_index_all)

if args.resume_checkpoint_path is not None:
    args.resume_model_path = os.path.join(args.resume_checkpoint_path, 'model-max.pth')
    args.resume_optimizer_path = os.path.join(args.resume_checkpoint_path, 'optimizer-max.pth')

model = ModelwithAttentionSupervision(model_path, layer_index_all, device, False, args.lora, args.bf16, args.model_type)
optimizer_parameters = []
for n, p in model.model.named_parameters():
    if p.requires_grad:
        optimizer_parameters.append(p)
optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr, weight_decay=args.weight_decay)

# load model and optimizer
if args.resume_model_path is not None:
    model_state_dict = torch.load(args.resume_model_path, map_location='cpu')
    model.model.load_state_dict(model_state_dict)
if args.resume_optimizer_path is not None:
    optimizer.load_state_dict(torch.load(args.resume_optimizer_path, map_location=torch.device('cpu')))

LLM_PREV_RESULTS_DIR = os.path.join(args.LLM_PREV_RESULTS_DIR, args.model_type)

# load questions
with open(os.path.join(DATA_DIR, 'task_items.pkl'), 'rb') as f:
    task_items = pickle.load(f)


# model prompt
prompt_syllogisms = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe a series of relationships among monosyllabic pseudowords and adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."
prompt_transitive = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe relationships among imaginary characters with comparative adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."

# get index
index_all = [i for i in range(70)]

# get train data
signal_dict = []
question_index = 0
# balance label class
num_true_all = 0
num_false_all = 0
for task_run in sorted(task_items.keys()):
    df = task_items[task_run]
    train_q = True
    if 'Transitive' in task_run:
        prompt = prompt_transitive
        if args.only_syll:
            train_q = False
    else:
        prompt = prompt_syllogisms
        if args.only_tran:
            train_q = False
    
    for i in range(len(df)):
        messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (df['premise1'][i].strip(), df['premise2'][i].strip(), df['premise3'][i].strip(), df['conclusion'][i].strip())},
        ]

        if 'Transitive' in task_run and '01' in task_run and (i == 7 or i == 13):
            continue
        else:
            # get label
            if 'true' in df['trial_type'][i]:
                label = 'True'
                # balance label class
                num_true_all += 1
            else:
                label = 'False'
                # balance label class
                num_false_all += 1
            if question_index in index_all and train_q:
                signal_dict.append({'x': messages, 'label': label})

            question_index += 1

if args.synthetic_data_file is not None:
    synthetic_data = load_from_json(args.synthetic_data_file)
    q_type_all = ['syllogisms', 'transitive']
    for q_type in q_type_all:
        if q_type == 'syllogisms':
            if args.only_tran:
                continue
            prompt = prompt_syllogisms
        if q_type == 'transitive':
            if args.only_syll:
                continue
            prompt = prompt_transitive
        
        for data in synthetic_data[q_type]:
            messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                    {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (data['premise1'].strip(), data['premise2'].strip(), data['premise3'].strip(), data['conclusion'].strip())},
            ]
            if 'true' in data['trial_type']:
                label = 'True'
                num_true_all += 1
            else:
                label = 'False'
                num_false_all += 1
            signal_dict.append({'x': messages, 'label': label})

batch_size = args.batch_size
# now only support batch size 1
assert batch_size == 1
grad_accumulate_step = args.grad_accumulate_step
if args.scheduler:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

# validation dataset
val_dataset = load_from_json(args.VAL_DATA)
val_acc_max = 0.
val_max_epoch = 0
# test dataset
test_dataset = load_from_json(args.TEST_DATA)

# training
current_iteration = 0
optimizer.zero_grad()
for e in range(args.epoch):
    if (e + 1) % 10 == 0:
        print('epoch ' + str(e))
    logging.info('epoch ' + str(e))
    model.model.train()

    iter_indices = list(range(len(signal_dict)))
    random.shuffle(iter_indices)

    total_loss = 0.
    # balance label class
    num_true = 0
    num_false = 0
    max_num_each = min(num_true_all, num_false_all)
    for idx in iter_indices:
        # get inputs
        sample = signal_dict[idx]
        x = sample['x']
        if args.model_type == 'llama2':
            text = model.tokenizer.apply_chat_template(x, tokenize=False)
            text += ' '
            encodeds = model.tokenizer([text], return_tensors='pt').to(device)
            input_ids = encodeds.input_ids
        if 'qwen3' in args.model_type:
            encodeds = model.tokenizer.apply_chat_template(x, return_tensors='pt', add_generation_prompt=True, enable_thinking=False)
            input_ids = encodeds.to(device)
        else:
            encodeds = model.tokenizer.apply_chat_template(x, return_tensors='pt', add_generation_prompt=True)
            input_ids = encodeds.to(device)
        attention_mask = None
        # get label info
        label = sample['label']
        label = model.tokenizer(label, return_tensors='pt').to(device)
        label = label['input_ids'].flatten()
        # avoid problem from mistral tokenizer
        label = label[-1:]

        if args.balance_label_class:
            q_label = sample['label']
            if q_label == 'True':
                num_true += 1
            else:
                num_false += 1
            if (q_label == 'True' and num_true > max_num_each) or (q_label == 'False' and num_false > max_num_each):
                continue

        if args.bf16:
            with autocast(dtype=torch.bfloat16):
                loss = model(input_ids, attention_mask, label=label) / grad_accumulate_step
        else:
            loss = model(input_ids, attention_mask, label=label) / grad_accumulate_step
        total_loss += loss.item()

        loss.backward()

        current_iteration += 1
        if current_iteration % grad_accumulate_step == 0:
            optimizer.step()
            optimizer.zero_grad()

    if args.balance_label_class:
        total_loss /= (max_num_each*2)
    else:
        total_loss /= len(signal_dict)
    total_loss *= grad_accumulate_step
    logging.info('loss {:.6f}'.format(total_loss))
    if args.scheduler:
        lr_scheduler.step()

    # validation
    if (e + 1) % args.val_epoch == 0:
        model.model.eval()
        if args.test_batch_size == 1:
            acc, syl_acc, tra_acc = test_model(model, val_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, parse_model_type, args.only_syll, args.only_tran, args.val_all_order)
        else:
            acc, syl_acc, tra_acc = test_model_batch(model, val_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, args.test_batch_size, parse_model_type, args.only_syll, args.only_tran, args.val_all_order)
        logging.info('--------------------')
        logging.info('Validation overall acc is: {:.3f}, syllogisms acc is: {:.3f}, transitive acc is: {:.3f}'.format(acc, syl_acc, tra_acc))
        logging.info('--------------------')
        if acc > val_acc_max:
            val_acc_max = acc
            val_max_epoch = e + 1
            save_model_path = os.path.join(RESULTS_DIR, 'model-max.pth')
            torch.save(model.model.state_dict(), save_model_path)
            save_optimizer_path = os.path.join(RESULTS_DIR, 'optimizer-max.pth')
            torch.save(optimizer.state_dict(), save_optimizer_path)

# use the best model on validation
logging.info(f'loading model from the epoch {val_max_epoch}')
save_model_path = os.path.join(RESULTS_DIR, 'model-max.pth')
model_state_dict = torch.load(save_model_path, map_location='cpu')
model.model.load_state_dict(model_state_dict)

if args.delete_checkpoint:
    # delete checkpoint
    os.remove(os.path.join(RESULTS_DIR, 'model-max.pth'))
    os.remove(os.path.join(RESULTS_DIR, 'optimizer-max.pth'))

# save model
save_model_path = os.path.join(RESULTS_DIR, 'model')
# merge lora
if args.lora:
    model.merge_lora()
model.model.save_pretrained(save_model_path)
model.tokenizer.save_pretrained(save_model_path)

# test set
if args.test_batch_size == 1:
    acc, syl_acc, tra_acc, correct_all = test_model(model, test_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, parse_model_type, False, False, True, return_results=True)
else:
    acc, syl_acc, tra_acc, correct_all = test_model_batch(model, test_dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, args.test_batch_size, parse_model_type, False, False, True, return_results=True)

path_results = os.path.join(RESULTS_DIR, f'test_behaviour_results.pkl')
with open(path_results, "wb") as f:
    pickle.dump(correct_all, f)

logging.info('--------------------')
logging.info('Test set')
logging.info('All order overall acc is: {:.3f}, syllogisms acc is: {:.3f}, transitive acc is: {:.3f}'.format(acc, syl_acc, tra_acc))

categories = [
    "2_true_affirm", "2_false_affirm",
    "3_true_affirm", "3_false_affirm",
    "2_true_negate", "2_false_negate",
    "3_true_negate", "3_false_negate"
]
trial_num_per_type = 100
num_order = 6
logging.info('Syllogisms:')
for i in range(len(categories)):
    idx_start = i * trial_num_per_type * num_order
    idx_end = (i + 1) * trial_num_per_type * num_order
    logging.info(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
logging.info('Transitive:')
for i in range(len(categories)):
    idx_start = (i + len(categories)) * trial_num_per_type * num_order
    idx_end = (i + 1 + len(categories)) * trial_num_per_type * num_order
    logging.info(categories[i] + ': {:.3f}'.format(average_list(correct_all[idx_start:idx_end])))
