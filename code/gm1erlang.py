import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(r'G:\My Drive\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')
from butools.ph import *
from butools.map import *
from butools.queues import *
import time
from butools.mam import *
from butools.dph import *
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
from tqdm import tqdm
from scipy.special import factorial
import os
import torch
import sympy
from sympy import *
from scipy.stats import gamma
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial
from datetime import datetime
import math

def compute_lst(x,prob_arr,ph):
    inv_mat = np.linalg.inv(x * np.identity(ph.shape[0]) - ph)
    first_lst = np.dot(prob_arr, inv_mat)
    ones = np.ones((ph.shape[0], 1))
    ph0 = -np.dot(ph, ones)
    lst = np.dot(first_lst, ph0)
    return lst

def ser_moment_n(s, A, mom):
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) *factorial(mom)*np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False


def gamma_pdf(x,theta,k):
    return (1/(gamma(k)))*(1/theta**k)*(np.exp(-x/theta))


def gamma_mfg(shape, scale, s):
    return (1-scale*s)**(-shape)


def get_nth_moment(shape, scale, n):
    s = Symbol('s')
    y = gamma_mfg(shape, scale, s)
    for i in range(n):
        if i == 0:
            dx = diff(y,s)
        else:
            dx = diff(dx,s)
    return dx.subs(s, 0)

def gamma_GM1_sample():

    try:

        rho = np.random.uniform(0.3,  0.99)
        shape = np.random.uniform(0.1, 3)
        scale = 1/(rho*shape)



        sigma = find_sigma(shape, scale)
        steady_gm1 = [1-rho]
        for l in range(1, 499):
            steady_gm1.append(rho*(1-sigma)*sigma**(l-1))
        steady_gm1 = np.append(steady_gm1,1-np.sum(np.array(steady_gm1)))


        moms_arr = np.array([])
        for mom in range(1, 21):
            moms_arr = np.append(moms_arr,np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))

        log_moms_arr  = np.log(moms_arr)

        exp1_moms_log = []
        for n in range(1, 21):
            exp1_moms_log.append(np.log(math.factorial(n)))

        x_data = np.append(log_moms_arr, np.array(exp1_moms_log[1:]))

        return (shape, scale, x_data, steady_gm1)
    except:
        print('Error')


def find_sigma(shape, scale):
    xvals = np.linspace(0,1,11)
    for val_ind, val in enumerate(xvals):
        if gamma_lst(1-val,shape, scale)<val:
            break
    xvals = np.linspace(xvals[val_ind-1],xvals[val_ind],450000)
    for val_ind, val in enumerate(xvals):
        if gamma_lst(1-val,shape, scale)<val:
            return val

def gamma_lst(s,shape, scale):
    return (1+shape*s)**(-scale)



def main():
    print('The current path is:')
    print(os.getcwd())

    if sys.platform == 'linux':

        if os.getcwd() == '/gpfs/fs0/scratch/d/dkrass/eliransc/Deep_queue/code':

            data_path = '/scratch/d/dkrass/eliransc/training/gm1'
        else:
            data_path = '/scratch/eliransc/training/gm1_gamma_1'

    else:

        data_path = r'C:\Users\user\workspace\data\deep_gm1'

    data_sample_name = 'batch_size_128_gamma_gm1_'


    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for i in range(1000):

        gm1_examples = [gamma_GM1_sample() for ind in tqdm(range(128))]

        gm1_examples = [example for example in gm1_examples if example]

        for ind, example in enumerate(gm1_examples):
            if ind == 0:
                moms_array = example[2].reshape(1, example[2].shape[0])
            else:
                moms_array = np.append(moms_array, example[2].reshape(1, example[2].shape[0]), axis=0)

        for ind, example in enumerate(gm1_examples):
            if ind == 0:
                ys_array = example[3].reshape(1, example[3].shape[0])
            else:
                ys_array = np.append(ys_array, example[3].reshape(1, example[3].shape[0]), axis=0)

        now = datetime.now()

        current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 10000000, 1)[0])

        pkl_name_moms = 'moms_' + str(20) + data_sample_name + current_time  + '.pkl'
        pkl_name_ys = 'y_' + str(20) + data_sample_name + current_time + '.pkl'

        pkl_name_moms = os.path.join(data_path, pkl_name_moms)
        pkl_name_ys = os.path.join(data_path, pkl_name_ys)

        moms_array = torch.tensor(moms_array).float()
        ys_array = torch.tensor(ys_array).float()

        pkl.dump(moms_array, open(pkl_name_moms, 'wb'))
        pkl.dump(ys_array,  open(pkl_name_ys, 'wb'))

        print('Iteration: ', i)






if __name__ == "__main__":
    main()

