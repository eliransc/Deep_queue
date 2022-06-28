from fastai.vision.all import *
# from fastbook import *
from sklearn.model_selection import train_test_split
matplotlib.rc('image', cmap='Greys')
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import torch
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
import torch
import torch.nn as nn

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
from scipy.special import factorial

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from datetime import datetime
m = nn.Softmax(dim=1)

def thresh_func(row):
    if (row['First_moment'] < mom_1_thresh) and (row['Second_moment'] < mom_2_thresh) and (
            row['Third_moment'] < mom_3_thresh):
        return True
    else:
        return False


def ser_mean(alph, T):
    e = np.ones((T.shape[0], 1))
    try:
        return -np.dot(np.dot(alph, np.linalg.inv(T)), e)
    except:
        return False


def compute_pdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return np.dot(np.dot(s, expm(A * x)), A0)


def compute_cdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return 1 - np.sum(np.dot(s, expm(A * x)))


def gives_rate(states_inds, rate, ph_size):
    '''
    states_ind: the out states indices
    rate: the total rate out
    return: the out rate array from that specific state
    '''
    final_rates = np.zeros(ph_size - 1)  ## initialize the array
    rands_weights_out_rate = np.random.rand(states_inds.shape[0])  ## Creating the weights of the out rate
    ## Computing the out rates
    final_rates[states_inds] = (rands_weights_out_rate / np.sum(rands_weights_out_rate)) * rate
    return final_rates


def create_row_rates(row_ind, is_absorbing, in_rate, non_abrosing_out_rates, ph_size, non_absorbing):
    '''
    row_ind: the current row
    is_abosboing: true if it an absorbing state
    in_rate: the rate on the diagonal
    non_abrosing_out_rates: the matrix with non_abrosing_out_rates
    ph_size: the size of phase type
    return: the ph row_ind^th of the ph matrix
    '''

    finarr = np.zeros(ph_size)
    finarr[row_ind] = -in_rate  ## insert the rate on the diagonal with a minus sign
    if is_absorbing:  ## no further changes is requires
        return finarr
    else:
        all_indices = np.arange(ph_size)
        all_indices = all_indices[all_indices != row_ind]  ## getting the non-diagonal indices
        rate_ind = np.where(non_absorbing == row_ind)  ## finding the current row in non_abrosing_out_rates
        finarr[all_indices] = non_abrosing_out_rates[rate_ind[0][0]]
        return finarr


def give_A_s_given_size(ph_size):
    potential_vals = np.linspace(0.5, 10, 20000)
    randinds = np.random.randint(potential_vals.shape[0], size=ph_size)
    ser_rates = (potential_vals[randinds]).reshape((1, ph_size))
    w = np.random.rand(ph_size + 1)
    numbers = np.arange(ph_size + 1)  # an array from 0 to ph_size + 1
    distribution = w / np.sum(w)  ## creating a pdf from the weights of w
    random_variable = rv_discrete(values=(numbers, distribution))  ## constructing a python pdf
    ww = random_variable.rvs(size=1)
    ## choosing the states that are absorbing
    absorbing_states = np.sort(np.random.choice(ph_size, ww[0], replace=False))
    non_absorbing = np.setdiff1d(np.arange(ph_size), absorbing_states, assume_unique=True)

    N = ph_size - ww[0]  ## N is the number of non-absorbing states
    p = np.random.rand()  # the probability that a non absorbing state is fully transient
    mask_full_trans = np.random.choice([True, False], size=N, p=[p, 1 - p])  # True if row sum to 0
    ser_rates = ser_rates.flatten()

    ## Computing the total out of state rate, if absorbing, remain the same
    p_outs = np.random.rand(N)  ### this is proportional rate out
    orig_rates = ser_rates[non_absorbing]  ## saving the original rates
    new_rates = orig_rates * p_outs  ## Computing the total out rates
    out_rates = np.where(mask_full_trans, orig_rates, new_rates)  ## Only the full trans remain as the original

    ## Choosing the number of states that will have a postive rate out for every non-absorbing state
    num_trans_states = np.random.randint(1, ph_size, N)

    ## Choosing which states will go from each non-absorbing state
    trans_states_list = [np.sort(np.random.choice(ph_size - 1, num_trans_states[j], replace=False)) for j in range(N)]
    # Computing out rates
    non_abrosing_out_rates = [gives_rate(trans_states, out_rates[j], ph_size) for j, trans_states in
                              enumerate(trans_states_list)]
    ## Finalizing the matrix

    #     return trans_states_list, absorbing_states, ser_rates, non_abrosing_out_rates
    lists_rate_mat = [
        create_row_rates(row_ind, row_ind in absorbing_states, ser_rates[row_ind], non_abrosing_out_rates, ph_size,
                         non_absorbing) for row_ind in range(ph_size)]
    A = np.concatenate(lists_rate_mat).reshape((ph_size, ph_size))  ## converting all into one numpy array

    num_of_pos_initial_states = np.random.randint(1, ph_size + 1)
    non_zero_probs = np.random.dirichlet(np.random.rand(num_of_pos_initial_states), 1)
    inds_of_not_zero_probs = np.sort(np.random.choice(ph_size, num_of_pos_initial_states, replace=False))
    s = np.zeros(ph_size)
    s[inds_of_not_zero_probs] = non_zero_probs

    return A, s


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def give_diff(v, i, ph_size):
    if i == 0:
        return v[i]
    elif i == len(v):
        return ph_size - v[-1]
    else:
        return v[i] - v[i - 1]


def generate_erlang(ind, ph_size, group):
    #     group = sequence[ind]

    A = np.identity(ph_size)
    s = np.zeros(ph_size)
    s[0] = 1
    #     ser_group_score = np.random.rand()
    #     if ser_group_score <1/3:
    #         group = 1
    #     elif ser_group_score>2/3:
    #         group = 3
    #     else:
    #         group = 2
    #     ser_rates = loguniform.rvs(1e-1, 10e2, size=1)
    ser_rates = np.random.uniform(0.5, 20)
    if group == 1:
        ser_rates = random.uniform(0.01, 0.1)
    elif group == 2:
        ser_rates = random.uniform(0.1, 0.5)
    else:
        ser_rates = random.uniform(1.0, 1.5)

    #     A = -A*ser_rates

    A_list = [create_erlang_row(ser_rates, ind, ph_size) for ind in range(ph_size)]
    A = np.concatenate(A_list).reshape((ph_size, ph_size))
    return (s, A)


def create_erlang_row(rate, ind, size):
    aa = np.zeros(size)
    aa[ind] = -rate
    if ind < size - 1:
        aa[ind + 1] = rate
    return aa


def ser_moment_n(s, A, mom):
    e = np.ones((A.shape[0], 1))
    try:
        mom = ((-1) ** mom) * factorial(mom) * np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom > 0:
            return mom
        else:
            return False
    except:
        return False


def get_rand_ph_dist(ph_size):
    potential_vals = np.linspace(0.1, 1, 2000)
    randinds = np.random.randint(potential_vals.shape[0], size=ph_size)
    ser_rates = (potential_vals[randinds]).reshape((1, ph_size))
    w = np.random.rand(ph_size + 1)
    numbers = np.arange(ph_size + 1)  # an array from 0 to ph_size + 1
    distribution = w / np.sum(w)  ## creating a pdf from the weights of w
    random_variable = rv_discrete(values=(numbers, distribution))  ## constructing a python pdf
    ww = random_variable.rvs(size=1)
    ## choosing the states that are absorbing
    absorbing_states = np.sort(np.random.choice(ph_size, ww[0], replace=False))
    non_absorbing = np.setdiff1d(np.arange(ph_size), absorbing_states, assume_unique=True)
    absorbing_states, non_absorbing

    N = ph_size - ww[0]  ## N is the number of non-absorbing states
    p = np.random.rand()  # the probability that a non absorbing state is fully transient
    mask_full_trans = np.random.choice([True, False], size=N, p=[p, 1 - p])  # True if row sum to 0
    ser_rates = ser_rates.flatten()

    ## Computing the total out of state rate, if absorbing, remain the same
    p_outs = np.random.rand(N)  ### this is proportional rate out
    orig_rates = ser_rates[non_absorbing]  ## saving the original rates
    new_rates = orig_rates * p_outs  ## Computing the total out rates
    out_rates = np.where(mask_full_trans, orig_rates, new_rates)  ## Only the full trans remain as the original

    ## Choosing the number of states that will have a postive rate out for every non-absorbing state
    num_trans_states = np.random.randint(1, ph_size, N)

    ## Choosing which states will go from each non-absorbing state
    trans_states_list = [np.sort(np.random.choice(ph_size - 1, num_trans_states[j], replace=False)) for j in range(N)]
    # Computing out rates
    non_abrosing_out_rates = [gives_rate(trans_states, out_rates[j], ph_size) for j, trans_states in
                              enumerate(trans_states_list)]
    ## Finalizing the matrix
    lists_rate_mat = [
        create_row_rates(row_ind, row_ind in absorbing_states, ser_rates[row_ind], non_abrosing_out_rates, ph_size,
                         non_absorbing) for row_ind in range(ph_size)]
    A = np.concatenate(lists_rate_mat).reshape((ph_size, ph_size))  ## converting all into one numpy array

    num_of_pos_initial_states = np.random.randint(1, ph_size + 1)
    non_zero_probs = np.random.dirichlet(np.random.rand(num_of_pos_initial_states), 1)
    inds_of_not_zero_probs = np.sort(np.random.choice(ph_size, num_of_pos_initial_states, replace=False))
    s = np.zeros(ph_size)
    s[inds_of_not_zero_probs] = non_zero_probs

    return s, A


def compute_first_n_moments(s, A, n=3):
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list


def is_all_n_moments_defined(mom_list):
    for mom in mom_list:
        if not mom:
            return False
    else:
        return True


def increase_size_of_matrix(s, A, max_ph_size):
    ph_size = A.shape[0]
    new_A = np.zeros((max_ph_size, max_ph_size))
    new_s = np.zeros(max_ph_size)
    new_A[:ph_size, :ph_size] = A
    new_s[:ph_size] = s
    new_A[(np.arange(ph_size, max_ph_size), np.arange(ph_size, max_ph_size))] = -1
    return new_s, new_A


def thresh_func(row):
    if (row['First_moment'] < mom_1_thresh) and (row['Second_moment'] < mom_2_thresh) and (
            row['Third_moment'] < mom_3_thresh):
        return True
    else:
        return False


def create_gen_erlang(sequenc, max_ph_size):
    ph_size = np.random.randint(3, max_ph_size + 1)
    num_samples_groups = min(ph_size, int(np.random.randint(2, ph_size) / 2))
    v = np.sort(np.random.choice(ph_size, num_samples_groups, replace=False))
    if v.shape[0] > 0:
        diff_list = np.array([give_diff(v, i, ph_size) for i in range(len(v) + 1)])
        diff_list = diff_list[diff_list > 0]

        erlang_list = [generate_erlang(ind, ph_size, sequenc[ind]) for ind, ph_size in enumerate(diff_list)]
        final_a = np.zeros((ph_size, ph_size))
        final_s = np.zeros(ph_size)
        rand_probs = np.random.dirichlet(np.random.rand(diff_list.shape[0]), 1)
        for ind in range(diff_list.shape[0]):
            final_s[np.sum(diff_list[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
            final_a[np.sum(diff_list[:ind]):np.sum(diff_list[:ind]) + diff_list[ind],
            np.sum(diff_list[:ind]):np.sum(diff_list[:ind]) + diff_list[ind]] = erlang_list[ind][1]
        return increase_size_of_matrix(final_s, final_a, max_ph_size)
    else:
        print('Not valid')


def compute_cdf_within_range(x_vals, s, A):
    pdf_list = []
    try:
        for x in x_vals:
            pdf_list.append(compute_cdf(x, s, A).flatten())
        return pdf_list
    except:
        print('Matrix goes to infinity')


def compute_pdf_within_range(x_vals, s, A):
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_pdf(x, s, A).flatten())

    return pdf_list


def recursion_group_size(group_left, curr_vector, phases_left):
    if group_left == 1:
        return np.append(phases_left, curr_vector)
    else:
        #         print(phases_left, group_left)
        if phases_left + 1 - group_left == 1:
            curr_size = 1
        else:
            curr_size = np.random.randint(1, phases_left + 1 - group_left)
        return recursion_group_size(group_left - 1, np.append(curr_size, curr_vector), phases_left - curr_size)


# def create_gen_erlang_given_sizes(group_sizes, rates, ph_size, probs = False):
#     erlang_list = [generate_erlang_given_rates(rates[ind], ph_size) for ind, ph_size in enumerate(group_sizes)]
#     final_a = np.zeros((ph_size, ph_size))
#     final_s = np.zeros(ph_size)
#     if not probs.any():
#         rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)
#         rands = np.random.rand(group_sizes.shape[0])
#         rand_probs = rands / np.sum(rands).reshape((1, rand_probs.shape[0]))
#     else:
#         rand_probs = probs
#     for ind in range(group_sizes.shape[0]):
#         final_s[np.sum(group_sizes[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
#         final_a[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
#         np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = erlang_list[ind]
#     return final_s, final_a


def generate_erlang_given_rates(rate, ph_size):
    A = np.identity(ph_size)
    A_list = [create_erlang_row(rate, ind, ph_size) for ind in range(ph_size)]
    A = np.concatenate(A_list).reshape((ph_size, ph_size))
    return A


def find_when_cdf_cross_1(x, y):
    if y[-1] < 0.9999:
        return False
    for ind in range(len(y)):
        if y[ind] > 0.9999:
            return ind
    return False


def find_normalizing_const(s, A, x, itera=0, thrsh=0.9999):
    #     print(itera)
    if itera > 50:
        return False
    curr_cdf = compute_cdf(x, s, A).flatten()[0]

    if curr_cdf < thrsh:
        return find_normalizing_const(s, A, x * 2, itera + 1, thrsh)
    elif (curr_cdf > thrsh) and (curr_cdf < 1.):
        return x
    else:
        return find_normalizing_const(s, A, x / 2, itera + 1, thrsh)


def normalize_matrix(s, A):
    normalize = find_normalizing_const(s, A, 6)
    if normalize > 1:
        A = A * normalize
    return (A, s)


def compute_R(lam, alph, T):
    e = torch.ones((T.shape[0], 1))
    return np.array(lam * torch.inverse(lam * torch.eye(T.shape[0]) - lam * e @ alph - T))


from numpy.linalg import matrix_power


def steady_i(rho, alph, R, i):
    return (1 - rho) * alph @ matrix_power(R, i)


def ser_mean(alph, T):
    e = torch.ones((T.shape[0], 1))
    try:
        return -alph @ torch.inverse(T) @ e
    except:
        return False


def combine_erlangs_lists(data_path, pkl_name, UB_ratios_limits, ph_size_max, UB_rates=1, LB_rates=0.1,
                          num_examples_each_settings=500):
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S") + str(np.random.randint(1, 1000000, 1)[0]) + '.pkl'
    pkl_name = pkl_name + current_time

    pkl_full_path = os.path.join(data_path, pkl_name)

    UB_ratios = np.random.randint(UB_ratios_limits[0], UB_ratios_limits[1])
    UB_rates = 1
    LB_rates = 0.1
    num_groups_max = int(ph_size_max / 2)

    x = generate_mix_ph(ph_size_max, np.random.randint(max(3, int(UB_ratios / 2)), max(4, UB_ratios)),
                        np.random.uniform(UB_rates, 2 * UB_rates), LB_rates,
                        np.random.randint(3, num_groups_max + 1))

    y_data = compute_y_data_given_folder(x, max_ph_size, tot_prob=70)

    x_y_data = (x, y_data)
    print(pkl_full_path)

    pkl.dump(x_y_data, open(pkl_full_path, 'wb'))

    return x_y_data


def generate_mix_ph(ph_size_max, UB_ratios, UB_rates, LB_rates, num_groups_max):
    num_groups = np.random.randint(2, num_groups_max + 1)
    ph_size = np.random.randint(num_groups, ph_size_max + 1)
    #     lam_arr = np.zeros((max_ph_size+1,1))

    group_sizes = recursion_group_size(num_groups, np.array([]), ph_size).astype(int)

    ratios = np.random.randint(1, UB_ratios, num_groups - 1)
    ratios = np.append(1, ratios)
    first_rate = np.random.uniform(LB_rates, UB_rates)
    rates = first_rate * ratios

    gen_erlang = create_gen_erlang_given_sizes(group_sizes, rates)

    A, s = normalize_matrix(gen_erlang[0], gen_erlang[1])

    s, A = increase_size_of_matrix(s, A, ph_size_max)

    if compute_cdf(1, s, A) < 0.999:
        return generate_mix_ph(ph_size_max, UB_ratios, UB_rates, LB_rates, num_groups_max)

    final_data = create_final_x_data(s, A, ph_size_max)

    return final_data


def create_final_x_data(s, A, ph_size_max):
    lam_arr = np.zeros((ph_size_max + 1, 1))

    s1 = s.reshape((1, s.shape[0]))
    expect_ser = ser_moment_n(s, A, 1)
    if expect_ser:
        #         expect_ser = expect_ser[0][0]
        lam = np.random.uniform(0, 1 / expect_ser, 1)[0]
        lam = lam * 0.95
        lam_arr[0, 0] = lam

        return np.append(np.append(A, s1, axis=0), lam_arr, axis=1).astype(np.float32)


def create_gewn_ph(ph_size_max, pkl_name, data_path):
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S") + str(np.random.randint(1, 1000000, 1)[0]) + '.pkl'
    pkl_name = pkl_name + current_time

    A_s_lists = [give_A_s_given_size(np.random.randint(2, ph_size_max)) for ind in range(1)]
    mom_lists = [compute_first_n_moments(tupl[1], tupl[0]) for tupl in A_s_lists]
    valid_list = [index for index, curr_mom_list in enumerate(mom_lists) if is_all_n_moments_defined(curr_mom_list)]
    normmat_ph_1 = [normalize_matrix(A_s[1], A_s[0]) for ind_A_s, A_s in enumerate(A_s_lists) if ind_A_s in valid_list]
    normmat_ph_1a = [(A_s[0], A_s[1]) for ind_A_s, A_s in enumerate(normmat_ph_1) if
                     compute_cdf(1, A_s[1], A_s[0]).flatten()[0] > 0.999]
    max_size_ph_1 = [increase_size_of_matrix(ph_dist[1], ph_dist[0], ph_size_max) for ph_dist in normmat_ph_1a]
    fin_data_reg = [create_final_x_data(ph_dist[0], ph_dist[1], ph_size_max) for ph_dist in max_size_ph_1]
    if len(fin_data_reg) > 0:
        x_y_data = compute_y_data_given_folder(fin_data_reg[0], ph_size_max, tot_prob=70)
        if type(x_y_data) == np.ndarray:
            pkl_full_path = os.path.join(data_path, pkl_name)
            pkl.dump((fin_data_reg[0], x_y_data), open(pkl_full_path, 'wb'))

            return (fin_data_reg[0], x_y_data)


def compute_y_data_given_folder(x, ph_size_max, tot_prob=70, eps=0.0001):
    try:
        lam = x[0, ph_size_max].item()
        A = x[:ph_size_max, :ph_size_max]
        s = x[ph_size_max, :ph_size_max].reshape((1, ph_size_max))
        expect_ser = ser_moment_n(s, A, 1)
        if expect_ser:
            rho = lam * expect_ser[0][0]

            R = compute_R(lam, s, A)

            steady_state = np.array([1 - rho])
            for i in range(1, tot_prob):
                steady_state = np.append(steady_state, np.sum(steady_i(rho, s, R, i)))
            if np.sum(steady_state) > 1 - eps:
                return steady_state
            else:
                return False

    except:
        print("x is not valid")


def create_short_tale(group_sizes, rates, probs):
    erlang_list = [generate_erlang_given_rates(rates[ind], ph_size) for ind, ph_size in enumerate(group_sizes)]
    ph_size = np.sum(group_sizes)
    final_a = np.zeros((ph_size, ph_size))
    final_s = np.zeros(ph_size)
    for ind in range(group_sizes.shape[0]):
        final_s[np.sum(group_sizes[:ind])] = probs[ind]
        final_a[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
        np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = erlang_list[ind]
    return final_s, final_a


def get_lower_upper_x(phsize, rate, prob_limit=0.999):
    lower_flag = False
    A = generate_erlang_given_rates(rate, phsize)
    s = np.zeros(phsize)
    s[0] = 1
    x_vals = np.linspace(0, 1, 300)
    for x in x_vals:
        if not lower_flag:
            if compute_cdf(x, s, A) > 0.001:
                lower = x
                lower_flag = True

        if compute_cdf(x, s, A) > prob_limit:
            upper = x

            return (lower, upper, phsize, rate)

    return False


import random


def give_rates_given_Er_sizes(df_, sizes, ratio_size):
    rates = np.array([])
    ratio_list = list(np.arange(ratio_size))
    for ph_size in sizes:
        curr_ratio = random.choice(ratio_list)
        ratio_list.remove(curr_ratio)
        inds = df_.loc[df_['phases'] == ph_size, :].index

        rates = np.append(rates, df_.loc[inds[curr_ratio], 'rate'])

    return rates


def create_rate_phsize_combs(vals_bound, ratios_rates):
    all_combs_list = []
    for size in vals_bound.keys():
        curr_list = [(size, vals_bound[size] * ratios_rates[ind_rate]) for ind_rate, rate in enumerate(ratios_rates)]
        all_combs_list.append(curr_list)
    return all_combs_list


def create_gen_erlang_given_sizes(group_sizes, rates, probs=False):
    ph_size = np.sum(group_sizes)
    erlang_list = [generate_erlang_given_rates(rates[ind], ph_size) for ind, ph_size in enumerate(group_sizes)]
    final_a = np.zeros((ph_size, ph_size))
    final_s = np.zeros(ph_size)
    if type(probs) == bool:
        rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)
        rands = np.random.rand(group_sizes.shape[0])
        rand_probs = rands / np.sum(rands).reshape((1, rand_probs.shape[0]))
    else:
        rand_probs = probs
    for ind in range(group_sizes.shape[0]):
        final_s[np.sum(group_sizes[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
        final_a[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
        np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = erlang_list[ind]
    return final_s, final_a


def find_upper_bound_rate_given_n(n, upper_bound):
    if upper_bound == 0:
        return False
    if np.array(get_lower_upper_x(n, upper_bound)).any():
        return find_upper_bound_rate_given_n(n, upper_bound - 1)
    else:
        return upper_bound + 1


def create_mix_erlang_data(df_1, max_num_groups=10, ratio_size=10):
    ph_sizes = np.linspace(10, 100, 10).astype(int)
    probs_ph_tot_size = np.array(ph_sizes ** 2 / np.sum(ph_sizes ** 2))
    num_groups = np.random.randint(1, max_num_groups + 1)
    group_size = recursion_group_size(num_groups, np.array([]),
                                      np.random.choice(ph_sizes, 1, p=probs_ph_tot_size)[0]).astype(int)
    group_rates = give_rates_given_Er_sizes(df_1, group_size, ratio_size)
    s, A = create_gen_erlang_given_sizes(group_size, group_rates)
    return (s, A)


def create_mix_erlang_data_steady(s, A, data_path, data_type, max_ph_size=100):
    now = datetime.now()

    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0]) + '.pkl'
    pkl_name = data_type + current_time

    pkl_full_path = os.path.join(data_path, pkl_name)

    A, s = normalize_matrix(s, A)

    s, A = increase_size_of_matrix(s, A, max_ph_size)

    final_data = create_final_x_data(s, A, max_ph_size)

    y_dat = compute_y_data_given_folder(final_data, max_ph_size, tot_prob=70)

    if type(y_dat) == bool:
        return False
        print('not dumping')
    else:
        pkl.dump((final_data, y_dat), open(pkl_full_path, 'wb'))
        return 1


def create_df_ph_bounds(vals_bounds_dict, ratios_rates):
    all_combs_list_1 = create_rate_phsize_combs(vals_bounds_dict, ratios_rates)

    merged_comb_list_1 = list(itertools.chain(*all_combs_list_1))

    df_list_1 = [get_lower_upper_x(phases_rates[0], phases_rates[1]) for phases_rates in tqdm(merged_comb_list_1)]

    lowerlist = []
    upperlist = []
    ph_size_list = []
    rate_list = []

    for item in tqdm(df_list_1):
        if item:
            lowerlist.append(item[0])
            upperlist.append(item[1])
            ph_size_list.append(item[2])
            rate_list.append(item[3])

    df_1 = pd.DataFrame(list(zip(lowerlist, upperlist, ph_size_list, rate_list)),
                        columns=['lower', 'upper', 'phases', 'rate'])

    pkl.dump(df_1, open('df_bound_ph.pkl', 'wb'))

def compute_m_y(data_path, sample_path):
    try:
        m_path = os.path.join(data_path, sample_path)
        y_path = os.path.join(data_path, 'ydat_'+sample_path[7:])
        m = pkl.load(open(m_path, 'rb'))
        y = pkl.load(open(y_path, 'rb'))
        return (m,y)
    except:
        print('No y data')


def train_epoch(model, lr, params):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()


def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')


def queue_loss(x, predictions, targes):
    predictions = m(predictions)
    #     aa = (x[:,0]*(torch.exp(x[:,1]))).reshape((x.shape[0],1))
    normalizing_const = 1 - targes[:, 0]
    predictions = predictions * normalizing_const.reshape(
        (targes.shape[0], 1))  # Normalizing the data such that it will sum to rho
    return ((torch.pow(torch.abs(predictions - targes[:, 1:]),1)).sum(axis = 1) + torch.max(torch.pow(torch.abs(predictions - targes[:, 1:]),1),1)[0]).sum()

def queue_loss1(x, predictions, targes):
    predictions = m(predictions)
    #     aa = (x[:,0]*(torch.exp(x[:,1]))).reshape((x.shape[0],1))
    normalizing_const = 1 - targes[:, 0]
    predictions = predictions * normalizing_const.reshape(
        (targes.shape[0], 1))  # Normalizing the data such that it will sum to rho
    return ((torch.abs(predictions - targes[:, 1:]))**0.5).sum() + torch.max((torch.abs(predictions - targes[:, 1:]))**0.5)


def valid_loss(dset_val, model):
    with torch.no_grad():
        for ind, batch in enumerate(dset_val):
            xb, yb = batch
            predictions = m(model(xb[:, :]))
            #             aa = (xb[:,0]*(torch.exp(xb[:,1]))).reshape((xb.shape[0],1))
            normalizing_const = 1 - yb[:, 0]  # aa.repeat(1,predictions.shape[1])
            predictions = predictions * normalizing_const.reshape((yb.shape[0], 1))
            curr_errors = torch.max(torch.abs((predictions - yb)), axis=1)[0]
            if ind == 0:
                max_err_tens = torch.max(torch.abs((predictions - yb)), axis=1)[0]
            else:
                max_err_tens = torch.cat((max_err_tens, curr_errors), axis=0)
        return torch.mean(max_err_tens)


def compute_sum_error(valid_dl, model, return_vector, max_err=0.05, display_bad_images=False):
    with torch.no_grad():
        bad_cases = {}
        for ind, batch in enumerate(valid_dl):

            xb, yb = batch
            predictions = m(model(xb[:, :]))
            #             aa = (xb[:,0]*(torch.exp(xb[:,1]))).reshape((xb.shape[0],1))
            normalizing_const = 1 - yb[:, 0]  # aa.repeat(1,predictions.shape[1])
            predictions = predictions * normalizing_const.reshape((yb.shape[0], 1))
            curr_errors = torch.sum(torch.abs((predictions - yb[:, 1:])), axis=1)
            bad_dists_inds = (curr_errors > max_err).nonzero(as_tuple=True)[0]
            # if display_bad_images:
            #     print_bad_examples(bad_dists_inds, predictions, yb)
            if ind == 0:
                sum_err_tens = torch.sum(torch.abs((predictions - yb[:, 1:])), axis=1)
            else:
                sum_err_tens = torch.cat((sum_err_tens, curr_errors), axis=0)
    if return_vector:
        return torch.mean(sum_err_tens), sum_err_tens
    else:
        return torch.mean(sum_err_tens)


def compute_max_error(valid_dl, model):

    with torch.no_grad():
        bad_cases = {}
        for ind, batch in enumerate(valid_dl):
            xb, yb = batch
            predictions = m(model(xb[:, :]))
            # aa = (xb[:,0]*(torch.exp(xb[:,1]))).reshape((xb.shape[0],1))
            normalizing_const = 1 - yb[:, 0]  # aa.repeat(1,predictions.shape[1])
            predictions = predictions * normalizing_const.reshape((yb.shape[0], 1))
            curr_errors = torch.sum(torch.abs((predictions - yb[:, 1:])), axis=1)  # [0]
            bad_error_inds = (curr_errors > 0.05).nonzero(as_tuple=True)[0]
            print(bad_error_inds)
            bad_cases[ind] = (bad_error_inds, xb, yb, ind, predictions, yb)
            if ind == 0:
                max_err_tens = torch.max(torch.abs((predictions - yb[:, 1:])), axis=1)[0]
            else:
                max_err_tens = torch.cat((max_err_tens, curr_errors), axis=0)
    return bad_cases


def valid(dset_val, model):
    loss = 0
    for batch in dset_val:
        xb, yb = batch
        loss += queue_loss(xb, model(xb), yb)
    return loss / len(dset_val)


def check_loss_increasing(loss_list, n_last_steps=10, failure_rate=0.45):
    counter = 0
    curr_len = len(loss_list)
    if curr_len < n_last_steps:
        n_last_steps = curr_len

    inds_arr = np.linspace(n_last_steps - 1, 1, n_last_steps - 1).astype(int)
    for ind in inds_arr:
        if loss_list[-ind] > loss_list[-ind - 1]:
            counter += 1

    print(counter, n_last_steps)
    if counter / n_last_steps > failure_rate:
        return True

    else:
        return False



def main():


    now = datetime.now()
    print('Start training')
    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])
    import torch
    ## Load data
    # mom_data_ = pkl.load(open('/scratch/eliransc/pkl_data/mom_data_7.pkl', 'rb'))
    # y_data_ = pkl.load(open('/scratch/eliransc/pkl_data/y_data_7.pkl', 'rb'))

    m_data = pkl.load(open('/scratch/eliransc/pkl_data/train_ph_moms.pkl', 'rb'))
    y_data = pkl.load(open('/scratch/eliransc/pkl_data/train_ph_ys.pkl', 'rb'))
    m_data_valid = pkl.load(open('/scratch/eliransc/pkl_data/gg1_mom_with_erlang_mg1_gm1_high_util_old_valid.pkl', 'rb'))
    y_data_valid = pkl.load(open('/scratch/eliransc/pkl_data/gg1_y_with_erlang_mg1_gm1_high_util_old_valid.pkl', 'rb'))

    # rates_valid = 1 / torch.exp(m_data_valid[:, 0])
    # rates_train = 1 / torch.exp(m_data[:, 0])
    bool_valid = y_data_valid[:, -1] > 0.01
    # bool_train = rates_train > 0.6

    m_data = m_data.float()
    y_data = y_data.float()
    m_data_valid = m_data_valid[bool_valid,:].float()
    y_data_valid = y_data_valid[bool_valid,:].float()

    for ind in range(m_data.shape[0]):
        if ind < 100:
            print(ind)

    # tot_vals = pkl.load(open('/scratch/eliransc/pkl_data/num_moms_vals.pkl', 'rb'))
    # nummom = tot_vals[0]
    # print(nummom)
    # tot_vals = tot_vals[1:]
    # pkl.dump(tot_vals, open('/scratch/eliransc/pkl_data/num_moms_vals.pkl', 'wb'))

    import time
    cur_time = int(time.time())
    data_path = '../new_gg1_models_4'
    seed = cur_time + len(os.listdir(data_path)) + np.random.randint(1, 1000)
    np.random.seed(seed)
    print(seed)

    archi = np.random.choice([1, 2, 3 ,4], size=1, replace=True, p=[0.25, 0.25, 0.25, 0.25])[0]
    bs = np.random.choice([64, 128], size=1, replace=True, p=[0.4, 0.6])[0]
    weight_deacy = np.random.choice([4, 5, 6], size=1, replace=True, p=[0.00, 0.99, 0.01])[0]
    num_moms_arrive = np.random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10], size=1, replace=True, p=[0.0, 0.0, 0.0, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])[0]
    num_moms_service = np.random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10], size=1, replace=True, p=[0.0, 0.0, 0.0, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])[0]
    lr_first = np.random.choice([0.7, 0.75, 0.8], size=1, replace=True, p=[0.3, 0.4, 0.3])[0]
    lr_second = np.random.choice([0.95, 0.98, 1], size=1, replace=True, p=[0.00, 0.3, 0.7])[0]

    print('The archi is: ', archi)
    print('The batch size is: ', bs)
    print('Weight decay is: ', weight_deacy)
    print('Number of arrival moments are: ', num_moms_arrive)
    print('Number of service moments are: ', num_moms_service)
    print('First lr: ', lr_first)
    print('Second lr: ', lr_second)

    setting_string = 'archi_' + str(archi) + '_bs_' + str(bs) + '_weight_decay_' + str(
        weight_deacy) + '_num_moms_arrival_' + str(num_moms_arrive) + '_num_moms_service_' + str(
        num_moms_service) + '_lr_first_' + str(lr_first) + '_lr_second_' + str(lr_second)




    now = datetime.now()

    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])
    print('curr time: ', current_time)


    # Construct dataset
    dset = list(zip(torch.cat((m_data[:, :num_moms_arrive], m_data[:, 20:20 + num_moms_service - 1]), 1), y_data))
    valid_dset = list(
        zip(torch.cat((m_data_valid[:, :num_moms_arrive], m_data_valid[:, 20:20 + num_moms_service - 1]), 1),
            y_data_valid))
    dl = DataLoader(dset, batch_size=bs)
    valid_dl = DataLoader(valid_dset, batch_size=bs)

    import torch
    import torch.nn as nn

    m = nn.Softmax(dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # code made in pytorch3.ipynb with comments
    # code made in pytorch3.ipynb with comments
    class Net(nn.Module):

        def __init__(self):
            super().__init__()

            if archi == 1:
                self.fc1 = nn.Linear(num_moms_arrive + num_moms_service - 1, 30)
                self.fc2 = nn.Linear(30, 50)
                self.fc3 = nn.Linear(50, 100)
                self.fc4 = nn.Linear(100, 200)
                self.fc5 = nn.Linear(200, 200)
                self.fc6 = nn.Linear(200, 350)
                self.fc7 = nn.Linear(350, 499)

            elif archi == 2:

                self.fc1 = nn.Linear(num_moms_arrive + num_moms_service - 1, 50)
                self.fc2 = nn.Linear(50, 70)
                self.fc3 = nn.Linear(70, 100)
                self.fc4 = nn.Linear(100, 200)
                self.fc5 = nn.Linear(200, 200)
                self.fc6 = nn.Linear(200, 350)
                self.fc7 = nn.Linear(350, 499)

            elif archi == 3:

                self.fc1 = nn.Linear(num_moms_arrive + num_moms_service - 1, 50)
                self.fc2 = nn.Linear(50, 70)
                self.fc3 = nn.Linear(70, 100)
                self.fc4 = nn.Linear(100, 150)
                self.fc5 = nn.Linear(150, 200)
                self.fc6 = nn.Linear(200, 200)
                self.fc7 = nn.Linear(200, 350)
                self.fc8 = nn.Linear(350, 499)

            else:

                self.fc1 = nn.Linear(num_moms_arrive + num_moms_service - 1, 30)
                self.fc2 = nn.Linear(30, 30)
                self.fc3 = nn.Linear(30, 30)
                self.fc4 = nn.Linear(30, 40)
                self.fc5 = nn.Linear(40, 50)
                self.fc6 = nn.Linear(50, 70)
                self.fc7 = nn.Linear(70, 100)
                self.fc8 = nn.Linear(100, 150)
                self.fc9 = nn.Linear(150, 200)
                self.fc10 = nn.Linear(200, 300)
                self.fc11 = nn.Linear(300, 400)
                self.fc12 = nn.Linear(400, 499)


        def forward(self, x):
            if archi == 3:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                x = F.relu(self.fc6(x))
                x = F.relu(self.fc7(x))
                x = self.fc8(x)
                return x

            elif archi == 4:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                x = F.relu(self.fc6(x))
                x = F.relu(self.fc7(x))
                x = F.relu(self.fc8(x))
                x = F.relu(self.fc9(x))
                x = F.relu(self.fc10(x))
                x = F.relu(self.fc11(x))
                x = self.fc12(x)
                return x


            else:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                x = F.relu(self.fc6(x))
                x = self.fc7(x)
                return x

    net = Net().to(device)

    curr_lr = 0.001

    dl.to(device)
    valid_dl.to(device)
    import time
    EPOCHS = 350

    optimizer = optim.Adam(net.parameters(), lr=curr_lr,
                           weight_decay=(1 / 10 ** weight_deacy))  # paramters is everything adjustable in model

    loss_list = []
    valid_list = []
    compute_sum_error_list = []

    for epoch in tqdm(range(EPOCHS)):
        t_0 = time.time()
        for data in dl:
            X, y = data

            net.zero_grad()
            output = net(X)
            loss = queue_loss(X, output, y)  # 1 of two major ways to calculate loss
            loss.backward()
            optimizer.step()
            net.zero_grad()

        loss_list.append(loss.item())
        valid_list.append(valid(valid_dl, net).item())
        compute_sum_error_list.append(compute_sum_error(valid_dl, net, False).item())

        if len(loss_list) > 3:
            if check_loss_increasing(valid_list):
                curr_lr = curr_lr * lr_first
                optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_deacy))
                print(curr_lr)
            else:
                curr_lr = curr_lr * lr_second
                optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_deacy))
                print(curr_lr)

        print("Epoch: {}, Training: {:.5f}, Validation : {:.5f}, Valid_sum_err: {:.5f},Time: {:.3f}".format(epoch,
                                                                                                            loss.item(),
                                                                                                            valid_list[
                                                                                                                -1],
                                                                                                            compute_sum_error_list[
                                                                                                                -1],
                                                                                                            time.time() - t_0))
        torch.save(net.state_dict(), '../new_gg1_models_4/pytorch_g_g_1_true_moms_new_data_' + setting_string + '_' + str(
            current_time) + '.pkl')
        pkl.dump((loss_list, valid_list, compute_sum_error_list),
                 open('../new_gg1_models_4/losts_' + '_new_data_' + setting_string + '_' + str(current_time) + '.pkl', 'wb'))


if __name__ == "__main__":

    main()