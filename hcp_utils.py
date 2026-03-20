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


# ==========================================
# New Utils for HCP Relational Task (Per-Subject Logic)
# ==========================================

def load_relational_llm_data(results_dir):
    """
    Load the dictionary based results for the relational task.
    Supports legacy-style {stim: {'attention': A, 'hidden': H}} entries.
    """
    path_activations = os.path.join(results_dir, 'hcp_relational_activations.pkl')
    path_behavior = os.path.join(results_dir, 'hcp_relational_behavior.pkl')
    
    print(f"Loading LLM activations from {path_activations}")
    with open(path_activations, "rb") as f:
        activations = pickle.load(f)
    print(f"Loading LLM behavior from {path_behavior}")
    with open(path_behavior, "rb") as f:
        behavior = pickle.load(f)
        
    return activations, behavior

def get_relational_llm_stimulus_index_map():
    """
    Build the LLM stimulus index map based on stimuli_mapping_final.jsonl order.
    Only keeps stimuli starting with 'int_' and ending with '.bmp'.
    """
    stimuli_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'stimuli_mapping_final.jsonl'))
    with open(stimuli_path, 'r') as f:
        items = [json.loads(line) for line in f]
    filtered = [it['stimulus_file'] for it in items
                if it['stimulus_file'].startswith('int_') and it['stimulus_file'].endswith('.bmp')]
    return {stim_name: idx for idx, stim_name in enumerate(filtered)}

def get_relational_fmri_subjects(fmri_results_dir, target_subjects=None):
    """
    Scan the FMRI_RESULTS_DIR for groups and subjects.
    Expected structure: FMRI_RESULTS_DIR / GroupName / SubjectName.mat
    """
    subjects_info = [] # List of dicts: {'group': g, 'subject': s, 'mat_path': p, 'csv_path': p_csv}
    
    # Check if dir exists
    if not os.path.exists(fmri_results_dir):
        print(f"Directory not found: {fmri_results_dir}")
        return subjects_info

    # Assuming Group folders are immediate children
    groups = sorted([d for d in os.listdir(fmri_results_dir) if os.path.isdir(os.path.join(fmri_results_dir, d))])
    
    print(f"Found groups: {groups}")

    for group in groups:
        group_path = os.path.join(fmri_results_dir, group)
        files = sorted([f for f in os.listdir(group_path) if f.endswith('.mat')])
        for f in files:
            subj_name = f.replace('.mat', '')
            if target_subjects is not None and subj_name not in target_subjects:
                continue
            mat_path = os.path.join(group_path, f)
            csv_path = os.path.join(group_path, subj_name + '_trialinfo.csv')
            
            if os.path.exists(csv_path):
                subjects_info.append({
                    'group': group,
                    'subject': subj_name,
                    'mat_path': mat_path,
                    'csv_path': csv_path
                })
            else:
                # Some .mat files might not be subjects (e.g. metadata), skip silently or warn
                pass
                
    return subjects_info

def align_subject_data(sub_info, llm_activations, llm_behavior, llm_stim_index_map=None):
    """
    Load fMRI data and align with LLM data for a single subject.
    """
    # Load fMRI
    try:
        mat_data = scipy.io.loadmat(sub_info['mat_path'])
    except Exception as e:
        print(f"Error loading {sub_info['mat_path']}: {e}")
        return None

    if 'all_voxels' not in mat_data:
        print(f"Key 'all_voxels' not found in {sub_info['mat_path']}")
        return None

    fmri_data = mat_data['all_voxels'] # N_trials x N_voxels
    
    # Load Meta
    try:
        df = pd.read_csv(sub_info['csv_path'])
    except Exception as e:
        print(f"Error loading {sub_info['csv_path']}: {e}")
        return None
    
    # Check lengths
    if len(df) != fmri_data.shape[0]:
        # Truncate to minimum length
        min_len = min(len(df), fmri_data.shape[0])
        df = df.iloc[:min_len]
        fmri_data = fmri_data[:min_len, :]
    
    # Build aligned lists
    aligned_llm_h = []
    model_correct = []
    human_correct = []
    valid_indices = []
    stimuli_info_list = []
    llm_question_indices = []
    
    for idx, row in df.iterrows():
        stim_name = str(row['Stimulus']).strip()
        
        # Match filename logic
        if stim_name not in llm_activations:
             # Try basics
             base = os.path.basename(stim_name)
             if base in llm_activations:
                 stim_name = base
             else:
                 # Try adding extension if missing
                 if not stim_name.endswith('.bmp'):
                      if stim_name + '.bmp' in llm_activations:
                          stim_name = stim_name + '.bmp'
        
        if stim_name in llm_activations:
            # Get LLM Data -> (Layers, Dim) or {'attention','hidden'} dict
            act = llm_activations[stim_name]
            if isinstance(act, dict):
                attn = act.get('attention')
                hid = act.get('hidden')
                if attn is None or hid is None:
                    # Skip malformed entries
                    continue
                act = np.concatenate([attn, hid], axis=0)
            beh = llm_behavior[stim_name]
            
            aligned_llm_h.append(act)
            model_correct.append(1 if beh['correct'] else 0)
            human_correct.append(int(row['ACC']))
            valid_indices.append(idx)
            stimuli_info_list.append(beh['stimulus_info'])
            if llm_stim_index_map is not None:
                llm_question_indices.append(llm_stim_index_map[stim_name])
        else:
            # print(f"Warning: stimulus {stim_name} not found in LLM results.")
            pass
       
    # If no matching trials, return None
    if len(aligned_llm_h) == 0:
        return None
        
    # Stack LLM: (N_samples, N_layers(=2*layer), N_dim)
    aligned_llm_h = np.stack(aligned_llm_h, axis=0) 
    
    # Filter fMRI to valid indices
    fmri_data_valid = fmri_data[valid_indices]
    
    return {
        'llm_rep': aligned_llm_h,
        'fmri_data': fmri_data_valid,
        'model_acc': np.array(model_correct),
        'human_acc': np.array(human_correct),
        'stimuli': df['Stimulus'].iloc[valid_indices].tolist(),
        'stimulus_info': stimuli_info_list,
        'llm_question_indices': llm_question_indices
    }


def construct_relational_prompt(item_info):
    """
    Constructs the prompt messages for the Relational Task based on item info.
    Supports dynamic attributes (Color, Style, etc.) if provided in 'active_attributes'.
    Fallbacks to Shape/Texture if no 'active_attributes' found.
    (Mirrors logic from hcp_get_activations.py)
    """
    # Mapping Definitions (Legacy)
    SHAPE_MAP = {
        1: "circle", 2: "cross", 3: "triangle", 4: "square", 5: "star", 6: "hexagon"
    }
    TEXTURE_MAP = {
        1: "ring_dots", 2: "angular_ticks", 3: "zigzag_dashes", 4: "solid_blocks", 5: "knot_weave", 6: "swirl_hooks"
    }

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
        
        # Determine value keys
        # If new format, keys are 'color', 'style' etc directly.
        # If old format, keys are 'shape_id', 'texture_id'.
        
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

    prompt_relational = (
        "You are an expert in relational reasoning. "
        "I will provide descriptions of four images (A, B, C, D), which describe two attributes of each image: "
        f"{attr1} and {attr2}. "
        "Image A and Image B differ in exactly one attribute "
        f"(either {attr1} or {attr2}). "
        "Your task is to determine whether Image C and Image D also differ in that attribute. "
        f"For example, if Image A and Image B have different {attr1}s, then you should determine if Image C and Image D also have different {attr1}s. "
        f"Or if Image A and Image B have different {attr2}s, then you should determine if Image C and Image D also have different {attr2}s. "
        "If the statement is true, you should answer 'True'; otherwise answer 'False'. "
        "Only answer 'True' or 'False' at the beginning of the response. "
    )
    
    stimulus_desc = (
        f"1. Image A: {get_desc('TopLeft')}.\n"
        f"2. Image B: {get_desc('TopRight')}.\n"
        f"3. Image C: {get_desc('BottomLeft')}.\n"
        f"4. Image D: {get_desc('BottomRight')}.\n"
        f"For the attribute that Image A and Image B differ in (either {attr1} or {attr2}), do Image C and Image D also differ in that attribute? Only answer 'True' or 'False' at the beginning of your response."
    )

    messages = [
            {"role": "user", "content": prompt_relational},
            {"role": "assistant", "content": "Understood. I will analyze the attributes and perform relational reasoning to judge the statement. I will only answer 'True' or 'False' at the beginning of the response."},
            {"role": "user", "content": stimulus_desc},
    ]
    return messages


def parse_relational_response(ans, model_type='default', warn_ambiguous=True):
    """
    Parses the LLM response for Relational Task (Yes/No).
    (Mirrors logic from hcp_get_activations.py)
    """
    # 1. DeepSeek Parsing
    if 'deepseek' in model_type or 'DeepSeek' in model_type: # Robust check
        idx = ans.rfind('</think>')
        if idx != -1:
            ans = ans[idx+len('</think>'):].strip('\n').strip()
        
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

    # 2. General Normalization
    ans_clean = ans.lower().strip().strip("'").strip('"').strip('.')
    model_ans_val = None
    
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
            if warn_ambiguous:
                print(f"Warning: ambiguous answer: {ans}")
            model_ans_val = -1 # Ambiguous
            
    return model_ans_val