import os
import pandas as pd
import pickle
import numpy as np
from utils import average_list, load_and_filter_behavior_results
import argparse

parser = argparse.ArgumentParser(description='Analyze behaviour results')
parser.add_argument('-DATA_DIR', type=str, default='./neuroimaging_info/') # path to get 'events.pkl' for human behaviour
parser.add_argument('-RESULTS_DIR', type=str, default='./results/activations_results/') # path to get model behaviour
parser.add_argument('-model_type', type=str, default='mistral') # path to get model behaviour

args = parser.parse_args()

DATA_DIR = args.DATA_DIR
RESULTS_DIR = os.path.join(args.RESULTS_DIR, args.model_type)

# filter these subjects due to missing / abnormal accuracy results or failure to normally process fMRI
filter_index = ['sub-1010', 'sub-1016', 'sub-1026', 'sub-1031', 'sub-1032', 'sub-1035', 'sub-1021']

# data_dict: {sub: {task_run}}, model_ans_all: {task_run}
_, data_dict, sub_list, model_ans_all, label_all, correct_all = load_and_filter_behavior_results(DATA_DIR, RESULTS_DIR, filter_index)

# calculate model accuracy
model_syllogisms_acc = []
model_transitive_acc = []
for task_run in sorted(correct_all.keys()):
    if 'Transitive' in task_run:
        model_transitive_acc += correct_all[task_run]
    else:
        model_syllogisms_acc += correct_all[task_run]
model_acc = model_syllogisms_acc + model_transitive_acc
print('Model accuracy overall is ' + str(average_list(model_acc)) + ', syllogisms is ' + str(average_list(model_syllogisms_acc)), ', transitive is ' + str(average_list(model_transitive_acc)))

# calculate human accuracy and model-human consistency
sub_task_acc = {}
overall_acc = []
syllogisms_acc = []
transitive_acc= []
overall_con_all = []
syllogisms_con_all = []
transitive_con_all = []
for sub in sub_list:
    sub_task_acc[sub] = {}
    sub_task_acc[sub]['syllogisms'] = []
    sub_task_acc[sub]['transitive'] = []
    print('--------------------')
    print(sub)

    for task_run in sorted(data_dict[sub]):
        sub_acc_all = list(data_dict[sub][task_run]['accuracy'])
        if 'Transitive' in task_run:
            sub_task_acc[sub]['transitive'] += sub_acc_all
        else:
            sub_task_acc[sub]['syllogisms'] += sub_acc_all
    sub_task_acc[sub]['overall'] = sub_task_acc[sub]['syllogisms'] + sub_task_acc[sub]['transitive']
    overall_acc.append(average_list(sub_task_acc[sub]['overall']))
    syllogisms_acc.append(average_list(sub_task_acc[sub]['syllogisms']))
    transitive_acc.append(average_list(sub_task_acc[sub]['transitive']))

    print(sub + ' accuracy overall is ' + str(overall_acc[-1]) + ', syllogisms is ' + str(syllogisms_acc[-1]) + ', transitive is ' + str(transitive_acc[-1]))

    overall_con = [x == y for x, y in zip(model_acc, sub_task_acc[sub]['overall'])]
    syllogisms_con = [x == y for x, y in zip(model_syllogisms_acc, sub_task_acc[sub]['syllogisms'])]
    transitive_con = [x == y for x, y in zip(model_transitive_acc, sub_task_acc[sub]['transitive'])]

    print(sub + ' consistency with model overall is ' + str(average_list(overall_con)) + ', syllogisms is ' + str(average_list(syllogisms_con)), ', transitive is ' + str(average_list(transitive_con)))

    overall_con_all.append(average_list(overall_con))
    syllogisms_con_all.append(average_list(syllogisms_con))
    transitive_con_all.append(average_list(transitive_con))

print('Human average accuracy overall is ' + str(average_list(overall_acc)) + ', syllogisms is ' + str(average_list(syllogisms_acc)), ', transitive is ' + str(average_list(transitive_acc)))
print('Human-model average consistency overall is ' + str(average_list(overall_con_all)) + ', syllogisms is ' + str(average_list(syllogisms_con_all)), ', transitive is ' + str(average_list(transitive_con_all)))

