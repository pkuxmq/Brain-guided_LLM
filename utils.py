import os
import pandas as pd
import pickle
import numpy as np
import scipy.io
from sklearn.linear_model import Ridge, RidgeCV
import scipy
from scipy.stats import pearsonr
import argparse
import math
import torch
import torch.nn as nn
import json
import random
from sklearn.decomposition import PCA
import itertools

# utils for calculation
def pearsonr_matrix(X, Y):
    # X, Y: n*d, n is the sample number for pearsonr, d is the parallel dimension
    mean_X = X.mean(axis=0, keepdims=True)
    mean_Y = Y.mean(axis=0, keepdims=True)

    mX = X - mean_X
    mY = Y - mean_Y

    normX = scipy.linalg.norm(mX, axis=0, keepdims=True)
    normX[normX==0] = 1
    normY = scipy.linalg.norm(mY, axis=0, keepdims=True)
    normY[normY==0] = 1

    r = ((mX / normX) * (mY / normY)).sum(axis=0)

    return r

def cosine_matrix(X, Y):
    # X, Y: n*d
    normX = scipy.linalg.norm(X, axis=1, keepdims=True)
    normX[normX==0] = 1
    normY = scipy.linalg.norm(Y, axis=1, keepdims=True)
    normY[normY==0] = 1

    r = ((X / normX) * (Y / normY)).sum(axis=1)

    return r

def calculate_corr(x_train, y_train, x_test, y_test, ridge_alpha=100.):
    model = Ridge(alpha=ridge_alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    corr = pearsonr_matrix(y_test, y_pred)
    return corr

def calculate_corr_cv(x_train, y_train, x_test, y_test, alphas=[1e-3, 1e-2, 1e-1, 1., 10., 1e2, 1e3]):
    model = RidgeCV(alphas=alphas)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    corr = pearsonr_matrix(y_test, y_pred)
    #print(model.alpha_)
    return corr

def calculate_cos_cv(x_train, y_train, x_test, y_test, alphas=[1e-3, 1e-2, 1e-1, 1., 10., 1e2, 1e3]):
    model = RidgeCV(alphas=alphas)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cos = cosine_matrix(y_test, y_pred)
    return cos.mean()

def calculate_corr_withW(x_test, y_test, W):
    y_pred = x_test @ W.T
    corr = pearsonr_matrix(y_test, y_pred)
    return corr

def calculate_mse_withW(x_test, y_test, W):
    y_pred = x_test @ W.T
    mse = np.mean((y_pred - y_test) ** 2) / 2
    return mse

def calculate_cosine_withW(x_test, y_test, W):
    y_pred = x_test @ W.T
    cosine_sim = cosine_matrix(y_test, y_pred)
    return cosine_sim

def mad(data, axis=0):
    median = np.median(data, axis=axis)
    abs_deviation = np.abs(data - median)
    mad_value = np.median(abs_deviation, axis=axis)
    return mad_value

def average_list(x_list):
    if len(x_list) == 0:
        return 0.
    return sum(x_list) * 1.0 / len(x_list)

def pearsonr_pytorch(X, Y):
    mean_X = torch.mean(X, dim=0, keepdim=True)
    mean_Y = torch.mean(Y, dim=0, keepdim=True)
    mX = X - mean_X
    mY = Y - mean_Y
    normX = torch.norm(mX, dim=0, keepdim=True)
    normY = torch.norm(mY, dim=0, keepdim=True)
    #normX[normX==0] = 1
    #normY[normY==0] = 1
    r = torch.sum((mX / normX) * (mY / normY), dim=0)
    return r

def normalize_vectors(v):
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms[norms==0] = 1
    return v / norms


# utils for loading data
def filter_questions(data_dict, model_ans_all, label_all, correct_all):
    # filter contradictory questions
    for task_run in sorted(model_ans_all.keys()):
        if 'Transitive' in task_run and '01' in task_run:
            del model_ans_all[task_run][13]
            del model_ans_all[task_run][7]
            del label_all[task_run][13]
            del label_all[task_run][7]
            del correct_all[task_run][13]
            del correct_all[task_run][7]

            for sub in sorted(data_dict.keys()):
                if task_run in data_dict[sub].keys():
                    data_dict[sub][task_run].drop(13, inplace=True)
                    data_dict[sub][task_run].drop(7, inplace=True)
    return data_dict, model_ans_all, label_all, correct_all

def load_question_sublist(DATA_DIR, filter_sub_index):
    # load questions
    with open(os.path.join(DATA_DIR, 'task_items.pkl'), 'rb') as f:
        # {task_run: {premise1, premise2, premise3, conclusion}}
        task_items = pickle.load(f)

    # load human behaviour results
    with open(os.path.join(DATA_DIR, 'events.pkl'), 'rb') as f:
        # {sub: {task_run}}
        data_dict = pickle.load(f)

    # get sub list
    sub_list = []
    for sub in sorted(data_dict.keys()):
        if sub in filter_sub_index:
            continue
        sub_list.append(sub)

    return task_items, data_dict, sub_list

def load_and_filter_behavior_results(DATA_DIR, RESULTS_DIR, filter_sub_index=[]):
    task_items, data_dict, sub_list = load_question_sublist(DATA_DIR, filter_sub_index)
    
    # load model behaviour results
    path_results = os.path.join(RESULTS_DIR, f'behaviour_results.pkl')
    with open(path_results, "rb") as f:
        # {task_run}
        [model_ans_all, label_all, correct_all] = pickle.load(f)

    data_dict, model_ans_all, label_all, correct_all = filter_questions(data_dict, model_ans_all, label_all, correct_all)
    return task_items, data_dict, sub_list, model_ans_all, label_all, correct_all

def get_model_human_behavior(correct_all, data_dict, sub_list, return_model_sep_acc=False):
    model_syllogisms_acc = []
    model_transitive_acc = []
    for task_run in sorted(correct_all.keys()):
        if 'Transitive' in task_run:
            model_transitive_acc += correct_all[task_run]
        else:
            model_syllogisms_acc += correct_all[task_run]
    model_acc = model_syllogisms_acc + model_transitive_acc

    sub_task_acc = {}
    for sub in sub_list:
        sub_task_acc[sub] = {}
        sub_task_acc[sub]['syllogisms'] = []
        sub_task_acc[sub]['transitive'] = []
        for task_run in sorted(data_dict[sub].keys()):
            sub_acc_all = list(data_dict[sub][task_run]['accuracy'])
            if 'Transitive' in task_run:
                sub_task_acc[sub]['transitive'] += sub_acc_all
            else:
                sub_task_acc[sub]['syllogisms'] += sub_acc_all
        sub_task_acc[sub]['overall'] = sub_task_acc[sub]['syllogisms'] + sub_task_acc[sub]['transitive']

    if return_model_sep_acc:
        return model_syllogisms_acc, model_transitive_acc, model_acc, sub_task_acc

    return model_acc, sub_task_acc

def load_LLM_features(results_dir, task_run_list):
    state_a_all = []
    state_h_all = []
    for task_run in sorted(task_run_list):
        path_h = os.path.join(results_dir, f'{task_run}_hidden.npy')
        path_a = os.path.join(results_dir, f'{task_run}_attention.npy')
        # 18 * nlayer * ndim
        state_h = np.load(path_h)
        # 18 * nlayer * nhead * ndim/nhead
        state_a = np.load(path_a)
        # remove contradictory questions
        if 'Transitive' in task_run and '01' in task_run:
            state_h = np.delete(state_h, 13, axis=0)
            state_h = np.delete(state_h, 7, axis=0)
            state_a = np.delete(state_a, 13, axis=0)
            state_a = np.delete(state_a, 7, axis=0)
    
        state_a_all.append(np.reshape(state_a, (state_a.shape[0], state_a.shape[1], -1)))
        state_h_all.append(state_h)
    # 70 * nlayer * ndim
    state_a_all = np.concatenate(state_a_all, axis=0)
    state_h_all = np.concatenate(state_h_all, axis=0)
    # N * nlayer*2 * ndim
    LLM_rep_all = np.concatenate([state_a_all, state_h_all], axis=1)
    return LLM_rep_all

def load_LLM_features_control(results_dir, task_run_list):
    state_a_all = []
    state_h_all = []
    for task_run in sorted(task_run_list):
        path_h = os.path.join(results_dir, f'{task_run}_hidden_control.npy')
        path_a = os.path.join(results_dir, f'{task_run}_attention_control.npy')
        # 18 * nlayer * ndim
        state_h = np.load(path_h)
        # 18 * nlayer * nhead * ndim/nhead
        state_a = np.load(path_a)
        # remove contradictory questions
        if 'Transitive' in task_run and '01' in task_run:
            state_h = np.delete(state_h, 13, axis=0)
            state_h = np.delete(state_h, 7, axis=0)
            state_a = np.delete(state_a, 13, axis=0)
            state_a = np.delete(state_a, 7, axis=0)
    
        state_a_all.append(np.reshape(state_a, (state_a.shape[0], state_a.shape[1], -1)))
        state_h_all.append(state_h)
    # 70 * nlayer * ndim
    state_a_all = np.concatenate(state_a_all, axis=0)
    state_h_all = np.concatenate(state_h_all, axis=0)
    # N * nlayer*2 * ndim
    LLM_rep_all = np.concatenate([state_a_all, state_h_all], axis=1)
    return LLM_rep_all

def get_all_fmri_data_latest(sub_list, FMRI_RESULTS_DIR):
    fmri_state_all = []
    for i, sub in enumerate(sub_list):
        path_fmri = os.path.join(FMRI_RESULTS_DIR, sub, f'{sub}.mat')
        fmri_state = scipy.io.loadmat(path_fmri)['all_voxels']
        # filter contradictory questions
        fmri_state = np.delete(fmri_state, 13+18*2, axis=0)
        fmri_state = np.delete(fmri_state, 7+18*2, axis=0)
        # 70 * fdim
        fmri_state_all.append(fmri_state)
    # 10 * 70 * fdim
    fmri_state_all = np.stack(fmri_state_all, axis=0)
    # filter nan
    nan_column = np.isnan(fmri_state_all).any(axis=(0, 1))
    fmri_state_all = fmri_state_all[:, :, ~nan_column]

    # syllogisms: 0, transitive: 1
    task_id = np.array([0] * 36 + [1] * 34)

    return fmri_state_all, task_id

def get_all_fmri_data_latest_PCA(sub_list, FMRI_RESULTS_DIR, pca_dim=500):
    fmri_state_all = []
    for i, sub in enumerate(sub_list):
        path_fmri = os.path.join(FMRI_RESULTS_DIR, sub, f'{sub}.mat')
        fmri_state = scipy.io.loadmat(path_fmri)['all_voxels']
        # filter contradictory questions
        fmri_state = np.delete(fmri_state, 13+18*2, axis=0)
        fmri_state = np.delete(fmri_state, 7+18*2, axis=0)
        # filter nan
        nan_column = np.isnan(fmri_state).any(axis=0)
        fmri_state = fmri_state[:, ~nan_column]
        # apply PCA
        pca = PCA(n_components=pca_dim)
        fmri_state = pca.fit_transform(fmri_state)
        # 70 * fdim
        fmri_state_all.append(fmri_state)
    # 10 * 70 * fdim
    fmri_state_all = np.stack(fmri_state_all, axis=0)

    # syllogisms: 0, transitive: 1
    task_id = np.array([0] * 36 + [1] * 34)

    return fmri_state_all, task_id


# utils for intervention
def get_intervention_info(LLM_rep_all, fmri_state, layer_index, fit_q_index=None, loss_type='mse', ridge_alpha=100., random=False, mask=None, return_W=False, noise_scale=None, use_std=True, random_fmri=False):
    # LLM_rep_all: attention + hidden, N * n_layer*2 * dim
    LLM_rep = LLM_rep_all[:, layer_index, :]
    if random_fmri:
        fmri_state = np.random.randn(*fmri_state.shape)
    
    model = Ridge(alpha=ridge_alpha)
    if fit_q_index is not None:
        model.fit(LLM_rep[fit_q_index], fmri_state[fit_q_index])
    else:
        model.fit(LLM_rep, fmri_state)
    W = model.coef_
    if return_W:
        W_ = W

    LLM_rep = torch.from_numpy(LLM_rep)
    rep_diff = torch.nn.Parameter(torch.zeros_like(LLM_rep))
    W = torch.from_numpy(W)
    fmri_state = torch.from_numpy(fmri_state).float()

    state_pred = torch.matmul((LLM_rep + rep_diff), W.T)
    if loss_type == 'pearsonr':
        r = pearsonr_pytorch(state_pred, fmri_state)
        if mask is not None:
            mask = torch.from_numpy(mask).float()
            r = r * mask
        loss = -torch.mean(r)
    elif loss_type == 'mse':
        mse_loss = torch.nn.MSELoss()
        if mask is not None:
            mask = torch.from_numpy(mask).float()
            loss = mse_loss(state_pred * mask, fmri_state * mask)
        else:
            loss = mse_loss(state_pred, fmri_state)
    else:
        raise("unsupported loss type")
    loss.backward()
    rep_dir = -rep_diff.grad.data.numpy()
    if random:
        rep_dir = np.random.randn(*rep_dir.shape)
    # normalize direction
    rep_dir = normalize_vectors(rep_dir)
    
    if noise_scale is not None:
        rep_dir_ = rep_dir
        # add noise
        rep_dir = rep_dir + np.random.randn(*rep_dir.shape) * noise_scale
        rep_dir = normalize_vectors(rep_dir)
        # size: sample_num
        cosine_sim = (rep_dir * rep_dir_).sum(axis=1)

    # get standard deviation for each direction
    proj_vals = LLM_rep.numpy() @ rep_dir.T
    std = np.std(proj_vals, axis=0, keepdims=True).T
    #print(std)
    #print(np.mean(std))
    if use_std:
        rep_dir *= std
    if noise_scale is not None:
        if return_W:
            return rep_dir, cosine_sim, W_, std
        return rep_dir, cosine_sim
    if return_W:
        return rep_dir, W_, std
    return rep_dir

def get_intervention_info_multistep(LLM_rep_all, fmri_state, layer_index, fit_q_index=None, intervene_index=None, loss_type='pearsonr', ridge_alpha=100., random=False, mask=None, return_W=False, noise_scale=None, use_std=True, random_fmri=False, lr=1e-4, iteration=100, save_per_iter=100, device='cuda'):
    # LLM_rep_all: attention + hidden, N * n_layer*2 * dim
    LLM_rep = LLM_rep_all[:, layer_index, :]
    if random_fmri:
        fmri_state = np.random.randn(*fmri_state.shape)
    
    model = Ridge(alpha=ridge_alpha)
    if fit_q_index is not None:
        model.fit(LLM_rep[fit_q_index], fmri_state[fit_q_index])
    else:
        model.fit(LLM_rep, fmri_state)
        fit_q_index = list(range(LLM_rep.shape[0]))

    W = model.coef_
    if return_W:
        W_ = W

    LLM_rep = torch.from_numpy(LLM_rep).to(device)
    rep_diff = torch.nn.Parameter(torch.zeros_like(LLM_rep))
    W = torch.from_numpy(W).to(device)
    fmri_state = torch.from_numpy(fmri_state).float().to(device)

    optimizer = torch.optim.AdamW([rep_diff], lr=lr)

    save_rep_diff = []
    for n_iter in range(iteration):
        state_pred = torch.matmul((LLM_rep + rep_diff), W.T)
        if intervene_index is not None:
            loss_index = list(set(fit_q_index + intervene_index))
        else:
            loss_index = list(range(LLM_rep.shape[0]))
        if loss_type == 'pearsonr':
            r = pearsonr_pytorch(state_pred[loss_index], fmri_state[loss_index])
            if mask is not None:
                mask = torch.from_numpy(mask).float().to(device)
                r = r * mask
            loss = -torch.mean(r)
        elif loss_type == 'mse':
            mse_loss = torch.nn.MSELoss()
            if mask is not None:
                mask = torch.from_numpy(mask).float().to(device)
                loss = mse_loss(state_pred * mask, fmri_state * mask)
            else:
                loss = mse_loss(state_pred, fmri_state)
        else:
            raise("unsupported loss type")
        #if n_iter % 100 == 0:
        #    print(loss)
        optimizer.zero_grad()
        loss.backward()
        if intervene_index is not None:
            grad_data = rep_diff.grad.data.clone()
            # normalize gradident for each question
            grad_data = grad_data / torch.norm(grad_data, dim=1, keepdim=True)
            rep_diff.grad.data = torch.zeros_like(grad_data)
            rep_diff.grad.data[intervene_index] = grad_data[intervene_index]
        optimizer.step()
        if (n_iter + 1) % save_per_iter == 0:
            save_rep_diff.append(rep_diff.data.cpu().numpy())
    #rep_dir = rep_diff.data.cpu().numpy()
    rep_dir = np.stack(save_rep_diff, axis=0)
    #print(rep_dir.shape)
    if return_W:
        return rep_dir, W_
    return rep_dir
#    rep_dir = -rep_diff.grad.data.numpy()
#    if random:
#        rep_dir = np.random.randn(*rep_dir.shape)
#    # normalize direction
#    rep_dir = normalize_vectors(rep_dir)
#    
#    if noise_scale is not None:
#        rep_dir_ = rep_dir
#        # add noise
#        rep_dir = rep_dir + np.random.randn(*rep_dir.shape) * noise_scale
#        rep_dir = normalize_vectors(rep_dir)
#        # size: sample_num
#        cosine_sim = (rep_dir * rep_dir_).sum(axis=1)
#
#    # get standard deviation for each direction
#    proj_vals = LLM_rep.numpy() @ rep_dir.T
#    std = np.std(proj_vals, axis=0, keepdims=True).T
#    #print(std)
#    #print(np.mean(std))
#    if use_std:
#        rep_dir *= std
#    if noise_scale is not None:
#        if return_W:
#            return rep_dir, cosine_sim, W_, std
#        return rep_dir, cosine_sim
#    if return_W:
#        return rep_dir, W_, std
#    return rep_dir

def get_intervention_info_multistep_each(LLM_rep_all, fmri_state, layer_index, fit_q_index=None, intervene_index=None, loss_type='pearsonr', ridge_alpha=100., random=False, mask=None, return_W=False, noise_scale=None, use_std=True, random_fmri=False, lr=1e-4, iteration=100, save_per_iter=100, device='cuda'):
    # LLM_rep_all: attention + hidden, N * n_layer*2 * dim
    LLM_rep = LLM_rep_all[:, layer_index, :]
    if random_fmri:
        fmri_state = np.random.randn(*fmri_state.shape)
    
    model = Ridge(alpha=ridge_alpha)
    if fit_q_index is not None:
        model.fit(LLM_rep[fit_q_index], fmri_state[fit_q_index])
    else:
        model.fit(LLM_rep, fmri_state)
    W = model.coef_
    if return_W:
        W_ = W

    LLM_rep = torch.from_numpy(LLM_rep).to(device)
    W = torch.from_numpy(W).to(device)
    fmri_state = torch.from_numpy(fmri_state).float().to(device)

    if intervene_index is None:
        intervene_index = list(range(LLM_rep.shape[0]))
    if fit_q_index is None:
        fit_q_index = list(range(LLM_rep.shape[0]))

    save_rep_diff = []
    rep_diff = torch.nn.Parameter(torch.zeros_like(LLM_rep))
    optimizer = torch.optim.AdamW([rep_diff], lr=lr)
    #optimizer = torch.optim.SGD([rep_diff], lr=lr, momentum=0.9)
    for n_iter in range(iteration):
        state_pred_ = torch.matmul(LLM_rep, W.T)
        optimizer.zero_grad()
        for ind in intervene_index:
            #state_pred = torch.matmul((LLM_rep + rep_diff), W.T)
            state_pred = state_pred_.clone().detach()
            state_pred[ind] = state_pred[ind] + torch.matmul(rep_diff[ind], W.T)
            loss_index = list(set(fit_q_index + [ind]))
            if loss_type == 'pearsonr':
                r = pearsonr_pytorch(state_pred[loss_index], fmri_state[loss_index])
                if mask is not None:
                    mask = torch.from_numpy(mask).float().to(device)
                    r = r * mask
                loss = -torch.mean(r)
            elif loss_type == 'mse':
                mse_loss = torch.nn.MSELoss()
                if mask is not None:
                    mask = torch.from_numpy(mask).float().to(device)
                    loss = mse_loss(state_pred * mask, fmri_state * mask)
                else:
                    loss = mse_loss(state_pred, fmri_state)
            else:
                raise("unsupported loss type")
            if n_iter % 100 == 0 and ind == intervene_index[0]:
                print('iter {:d}: {:.4f}'.format(n_iter, loss))
            loss.backward()
            # normalize gradient for each question
            grad_norm = torch.norm(rep_diff.grad.data[ind])
            #if n_iter == 0:
            #    print(grad_norm)
            rep_diff.grad.data[ind] = rep_diff.grad.data[ind] / grad_norm
            #if intervene_index is not None:
            #    grad_data = rep_diff.grad.data.clone()
            #    rep_diff.grad.data = torch.zeros_like(grad_data)
            #    rep_diff.grad.data[intervene_index] = grad_data[intervene_index]
        optimizer.step()
        if (n_iter + 1) % save_per_iter == 0:
            save_rep_diff.append(rep_diff.data.cpu().numpy())
        #rep_dir[ind] = rep_diff.data.cpu().numpy()
    #rep_dir = rep_diff.data.cpu().numpy()
    rep_dir = np.stack(save_rep_diff, axis=0)
    if return_W:
        return rep_dir, W_
    return rep_dir



def get_intervention_info_with_MLP(LLM_rep_all, fmri_state, layer_index, fit_q_index=None, mask_weight=None, mlp_layer_num=3, mlp_dim=4096, train_epoch=1000, train_batch=8, lr=1e-5, device='cuda', random=False, noise_scale=None, use_std=True, random_fmri=False):
    # LLM_rep_all: attention + hidden, N * n_layer*2 * dim
    LLM_rep = LLM_rep_all[:, layer_index, :]
    if random_fmri:
        fmri_state = np.random.randn(*fmri_state.shape)
    
    MLP = get_MLP_info(LLM_rep, fmri_state, fit_q_index, mask_weight, mlp_layer_num, mlp_dim, train_epoch, train_batch, lr, device)

    LLM_rep = torch.from_numpy(LLM_rep).to(device)
    rep_diff = torch.nn.Parameter(torch.zeros_like(LLM_rep)).to(device)
    fmri_state = torch.from_numpy(fmri_state).float().to(device)

    state_pred = MLP(LLM_rep + rep_diff)
    mse_loss = torch.nn.MSELoss()
    if mask_weight is not None:
        mask_weight = torch.from_numpy(mask_weight).float().to(device)
        loss = mse_loss(state_pred * mask_weight, fmri_state * mask_weight)
    else:
        loss = mse_loss(state_pred, fmri_state)
    loss.backward()
    rep_dir = -rep_diff.grad.data.cpu().numpy()
    if random:
        rep_dir = np.random.randn(*rep_dir.shape)
    # normalize direction
    rep_dir = normalize_vectors(rep_dir)
    
    if noise_scale is not None:
        rep_dir_ = rep_dir
        # add noise
        rep_dir = rep_dir + np.random.randn(*rep_dir.shape) * noise_scale
        rep_dir = normalize_vectors(rep_dir)
        # size: sample_num
        cosine_sim = (rep_dir * rep_dir_).sum(axis=1)

    # get standard deviation for each direction
    proj_vals = LLM_rep.cpu().numpy() @ rep_dir.T
    std = np.std(proj_vals, axis=0, keepdims=True).T
    if use_std:
        rep_dir *= std
    if noise_scale is not None:
        return rep_dir, MLP, cosine_sim
    return rep_dir, MLP

def get_intervention_info_unnormalized(LLM_rep_all, fmri_state, layer_index, fit_q_index=None, loss_type='mse', ridge_alpha=100., random=False):
    # LLM_rep_all: attention + hidden, N * n_layer*2 * dim
    LLM_rep = LLM_rep_all[:, layer_index, :]
    
    model = Ridge(alpha=ridge_alpha)
    if fit_q_index is not None:
        model.fit(LLM_rep[fit_q_index], fmri_state[fit_q_index])
    else:
        model.fit(LLM_rep, fmri_state)
    W = model.coef_

    LLM_rep = torch.from_numpy(LLM_rep)
    rep_diff = torch.nn.Parameter(torch.zeros_like(LLM_rep))
    W = torch.from_numpy(W)
    fmri_state = torch.from_numpy(fmri_state).float()

    state_pred = torch.matmul((LLM_rep + rep_diff), W.T)
    if loss_type == 'pearsonr':
        r = pearsonr_pytorch(state_pred, fmri_state)
        loss = -torch.mean(r)
    elif loss_type == 'mse':
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(state_pred, fmri_state)
    else:
        raise("unsupported loss type")
    loss.backward()
    rep_dir = -rep_diff.grad.data.numpy()
    if random:
        rep_dir = np.random.randn(*rep_dir.shape)
    return rep_dir

def get_intervention_info_withW(LLM_rep, fmri_state, W, loss_type='mse', std=None, mask=None):
    LLM_rep = torch.from_numpy(LLM_rep)
    rep_diff = torch.nn.Parameter(torch.zeros_like(LLM_rep))
    W = torch.from_numpy(W)
    fmri_state = torch.from_numpy(fmri_state).float()

    state_pred = torch.matmul((LLM_rep + rep_diff), W.T)
    if loss_type == 'pearsonr':
        r = pearsonr_pytorch(state_pred, fmri_state)
        if mask is not None:
            mask = torch.from_numpy(mask).float()
            r = r * mask
        loss = -torch.mean(r)
    elif loss_type == 'mse':
        mse_loss = torch.nn.MSELoss()
        if mask is not None:
            mask = torch.from_numpy(mask).float()
            loss = mse_loss(state_pred * mask, fmri_state * mask)
        else:
            loss = mse_loss(state_pred, fmri_state)
    else:
        raise("unsupported loss type")
    loss.backward()
    rep_dir = -rep_diff.grad.data.numpy()
    # normalize direction
    rep_dir = normalize_vectors(rep_dir)

    if std is None:
        # get standard deviation for each direction
        proj_vals = LLM_rep.numpy() @ rep_dir.T
        std = np.std(proj_vals, axis=0, keepdims=True).T
    rep_dir *= std
    return rep_dir

def get_W_info(LLM_rep_all, fmri_state, layer_index, fit_q_index=None, loss_type='mse', ridge_alpha=100., random=False, return_b=False):
    # LLM_rep_all: attention + hidden, N * n_layer*2 * dim
    LLM_rep = LLM_rep_all[:, layer_index, :]
    
    model = Ridge(alpha=ridge_alpha)
    if fit_q_index is not None:
        model.fit(LLM_rep[fit_q_index], fmri_state[fit_q_index])
    else:
        model.fit(LLM_rep, fmri_state)
    W = model.coef_

    if return_b:
        b = model.intercept_
        return W, b
    return W

def get_W_info_cv(LLM_rep_all, fmri_state, layer_index, fit_q_index=None, loss_type='mse', alphas=[1e-3, 1e-2, 1e-1, 1., 10., 1e2, 1e3], random=False, return_b=False):
    # LLM_rep_all: attention + hidden, N * n_layer*2 * dim
    LLM_rep = LLM_rep_all[:, layer_index, :]
    
    model = RidgeCV(alphas=alphas)
    #model = RidgeCV(alphas=alphas, fit_intercept=False)
    if fit_q_index is not None:
        model.fit(LLM_rep[fit_q_index], fmri_state[fit_q_index])
    else:
        model.fit(LLM_rep, fmri_state)
    W = model.coef_

    if return_b:
        b = model.intercept_
        return W, b

    return W

def get_MLP_info(LLM_rep, fmri_state, fit_q_index=None, mask_weight=None, mlp_layer_num=3, mlp_dim=4096, train_epoch=10, train_batch=8, lr=1e-4, device='cuda'):
    LLM_rep = torch.from_numpy(LLM_rep)
    fmri_state = torch.from_numpy(fmri_state).float()
    r_dim = LLM_rep.shape[1]
    f_dim = fmri_state.shape[1]

    layers = []
    layers += [nn.Linear(r_dim, mlp_dim), nn.SiLU()]
    for _ in range(mlp_layer_num - 2):
        layers += [nn.Linear(mlp_dim, mlp_dim), nn.SiLU()]
    layers += [nn.Linear(mlp_dim, f_dim)]
    MLP = nn.Sequential(*layers)
    MLP.to(device)
    optimizer = torch.optim.AdamW(MLP.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()

    # train
    if fit_q_index is None:
        indices = list(range(LLM_rep.shape[0]))
    else:
        indices = fit_q_index
    if mask_weight is not None:
        mask_weight = torch.from_numpy(mask_weight).float().to(device)
    print('MLP training')
    for e in range(train_epoch):
        random.shuffle(indices)
        iter_num = (len(indices) - 1) // train_batch + 1
        total_loss = 0
        for i in range(iter_num):
            iter_indices = indices[i * train_batch : min((i + 1) * train_batch, len(indices))]
            x, y = LLM_rep[iter_indices], fmri_state[iter_indices]
            x = x.to(device)
            y = y.to(device)
            if mask_weight is not None:
                loss = mse_loss(MLP(x) * mask_weight, y * mask_weight)
            else:
                loss = mse_loss(MLP(x), y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        total_loss /= iter_num
        if e == 0 or (e + 1) % 100 == 0:
            print('epoch {:d} loss {:.3f}'.format(e, total_loss))

    return MLP


# utils for fine-tune
def load_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)  # Load and parse the JSON data
    return data

def test_model(model, dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, parse_model_type='default', only_syll=False, only_tran=False, all_order=False, num_premises=3, return_results=False):
    correct_all = []
    q_type_all = ['syllogisms', 'transitive']
    for q_type in q_type_all:
        if q_type == 'syllogisms':
            if only_tran:
                continue
            prompt = prompt_syllogisms
        if q_type == 'transitive':
            if only_syll:
                continue
            prompt = prompt_transitive

        for data in dataset[q_type]:
            #premises = [data['premise1'].strip(), data['premise2'].strip(), data['premise3'].strip()]
            premises = []
            for i_p in range(num_premises):
                str_p = 'premise' + str(i_p + 1)
                premises.append(data[str_p].strip())
            if all_order:
                #premises_all_order = [premises, [premises[0], premises[2], premises[1]], [premises[1], premises[0], premises[2]], [premises[1], premises[2], premises[0]], [premises[2], premises[0], premises[1]], [premises[2], premises[1], premises[0]]]
                premises_all_order = [list(x) for x in list(itertools.permutations(premises))]
            else:
                premises_all_order = [premises]
            for premises in premises_all_order:
                messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                        {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (premises[0], premises[1], premises[2], data['conclusion'].strip())},
                ]
        
                with torch.no_grad():
                    ans = model.generate_ans(messages, max_new_tokens=max_new_tokens, parse_model_type=parse_model_type)
                #print(data['trial_type'] + ', ' + ans)
                if ans in ['True', 'true', 'True.']:
                    model_ans = 1
                elif ans in ['False', 'false', 'False.']:
                    model_ans = 0
                else:
                    print('No valid answer. The output is ' + ans)
                    model_ans = None
        
                trial_type = data['trial_type']
                if 'true' in trial_type:
                    label = 1
                else:
                    label = 0
                correct_all.append(model_ans==label)
    
    acc = average_list(correct_all)
    if only_syll:
        return acc, acc, -1
    if only_tran:
        return acc, -1, acc
    acc, syllogisms_acc, transitive_acc = average_list(correct_all), average_list(correct_all[:len(dataset['syllogisms'])*len(premises_all_order)]), average_list(correct_all[len(dataset['syllogisms'])*len(premises_all_order):])
    if return_results:
        return acc, syllogisms_acc, transitive_acc, correct_all
    return acc, syllogisms_acc, transitive_acc


def test_model_batch(model, dataset, prompt_syllogisms, prompt_transitive, max_new_tokens, batch_size, parse_model_type='default', only_syll=False, only_tran=False, all_order=False, num_premises=3, return_results=False):
    correct_all = []
    q_type_all = ['syllogisms', 'transitive']
    for q_type in q_type_all:
        if q_type == 'syllogisms':
            if only_tran:
                continue
            prompt = prompt_syllogisms
        if q_type == 'transitive':
            if only_syll:
                continue
            prompt = prompt_transitive

        cnt = 0
        batch_messages_list = []
        batch_label_list = []
        for data in dataset[q_type]:
            premises = []
            for i_p in range(num_premises):
                str_p = 'premise' + str(i_p + 1)
                premises.append(data[str_p].strip())
            if all_order:
                premises_all_order = [list(x) for x in list(itertools.permutations(premises))]
            else:
                premises_all_order = [premises]
            for premises in premises_all_order:
                messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                        {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (premises[0], premises[1], premises[2], data['conclusion'].strip())},
                ]
                trial_type = data['trial_type']
                if 'true' in trial_type:
                    label = 1
                else:
                    label = 0
                batch_messages_list.append(messages)
                batch_label_list.append(label)
                cnt += 1
        
                if cnt == batch_size:
                    with torch.no_grad():
                        batch_ans = model.generate_ans_batch(batch_messages_list, max_new_tokens=max_new_tokens, parse_model_type=parse_model_type)
                    for i,ans in enumerate(batch_ans):
                        if ans in ['True', 'true', 'True.']:
                            model_ans = 1
                        elif ans in ['False', 'false', 'False.']:
                            model_ans = 0
                        else:
                            model_ans = None
        
                        correct_all.append(model_ans==batch_label_list[i])
                    cnt = 0
                    batch_messages_list = []
                    batch_label_list = []
    
    acc = average_list(correct_all)
    if only_syll:
        return acc, acc, -1
    if only_tran:
        return acc, -1, acc
    acc, syllogisms_acc, transitive_acc = average_list(correct_all), average_list(correct_all[:len(dataset['syllogisms'])*len(premises_all_order)]), average_list(correct_all[len(dataset['syllogisms'])*len(premises_all_order):])
    if return_results:
        return acc, syllogisms_acc, transitive_acc, correct_all
    return acc, syllogisms_acc, transitive_acc


# utils for visualization
def get_LLM_rep(dataset, model, num_per_type):
    """Get LLM representations for deductive reasoning tasks"""
    prompt_syllogisms = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe a series of relationships among monosyllabic pseudowords and adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."
    prompt_transitive = "You are an expert in performing logical reasoning. I will present three premises and one conclusion each round. The premises describe relationships among imaginary characters with comparative adjectives. Assume all premises are True. You should use deductive reasoning to determine if the conclusion can be drawn from the premises logically. Answer 'True' if the conclusion can be drawn logically; otherwise you should answer 'False'. Only answer 'True' or 'False' at the beginning of the response."

    q_type_all = ['syllogisms', 'transitive']
    labels = []
    all_state_h = []
    all_state_a = []
    
    for q_type in q_type_all:
        if q_type == 'syllogisms':
            prompt = prompt_syllogisms
        if q_type == 'transitive':
            prompt = prompt_transitive

        cnt_per_type = {}
        for data in dataset[q_type]:
            trial_type = data['trial_type']
            if trial_type in cnt_per_type.keys():
                if cnt_per_type[trial_type] >= num_per_type:
                    continue
                cnt_per_type[trial_type] += 1
            else:
                cnt_per_type[trial_type] = 1
            
            # get label
            if 'true_affirm' in trial_type:
                labels.append(q_type + '_affirm_true')
            elif 'false_affirm' in trial_type:
                labels.append(q_type + '_affirm_false')
            elif 'true_negate' in trial_type:
                labels.append(q_type + '_negate_true')
            elif 'false_negate' in trial_type:
                labels.append(q_type + '_negate_false')
            
            # get representation
            premises = [data['premise1'].strip(), data['premise2'].strip(), data['premise3'].strip()]
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Understood. I will do my best to determine if the conclusion can be logically drawn from the given premises, and only answer 'True' or 'False' at the beginning of the response."},
                {"role": "user", "content": "Premises:\n1. %s.\n2. %s.\n3. %s.\nConclusion:\n%s." % (premises[0], premises[1], premises[2], data['conclusion'].strip())},
            ]
            state_h, state_a = model.get_all_states(messages)
            # last token
            state_h = state_h[:,-1]
            state_a = state_a[:,-1]
            # nlayer*nhead*head_dim => nlayer*ndim
            state_a = np.reshape(state_a, (state_a.shape[0], -1))
            all_state_h.append(state_h)
            all_state_a.append(state_a)
    
    # N * nlayer * ndim
    all_state_h = np.stack(all_state_h, axis=0)
    all_state_a = np.stack(all_state_a, axis=0)
    LLM_rep_all = np.concatenate([all_state_a, all_state_h], axis=1)
    print(f"Representations shape: {LLM_rep_all.shape}")
    print(f"Number of labels: {len(labels)}")
    return LLM_rep_all, labels