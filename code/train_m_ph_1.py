from fastai.vision.all import *
from fastbook import *
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

def compute_R(lam, alph, T):
    e = torch.ones((T.shape[0],1))
    return lam*torch.inverse(lam*torch.eye(T.shape[0]) - lam*e@alph - T)
from numpy.linalg import matrix_power
def steady_i(rho,  alph, R, i):
    return (1-rho)*alph@matrix_power(R,i)
def ser_mean(alph, T):
    e = torch.ones((T.shape[0],1))
    try:
        return -alph@torch.inverse(T)@e
    except: return False


def ser_moment_n(s, A, mom):
    e = np.ones((A.shape[0], 1))
    try:
        mom = ((-1) ** mom) * np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom > 0:
            return mom
        else:
            return False
    except:
        return False


def thresh_func(row):
    if (row['First_moment'] < mom_1_thresh) and (row['Second_moment'] < mom_2_thresh) and (
            row['Third_moment'] < mom_3_thresh):
        return True
    else:
        return False


def compute_pdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return np.dot(np.dot(s, expm(A * x)), A0)


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


def compute_R(lam, alph, T):
    alph = np.array(alph)
    T = np.array(T)
    e = np.ones((T.shape[0], 1))
    return torch.tensor(lam * np.linalg.inv(lam * np.identity(T.shape[0]) - lam * np.dot(e, alph) - T))


from numpy.linalg import matrix_power


def steady_i(rho, alph, R, i):
    return torch.tensor((1 - rho) * np.dot(alph, matrix_power(R, i)))


def ser_mean(alph, T):
    e = np.ones((T.shape[0], 1))
    try:
        return -np.dot(np.dot(alph, np.linalg.inv(T)), e)
    except:
        return False


def compute_pdf_within_range(x_vals, s, A):
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_pdf(x, s, A).flatten())

    return pdf_list


def getX(pkl_path, sample_path):
    curr_data = pkl.load(open(os.path.join(pkl_path, sample_path),'rb'))
    data = torch.from_numpy(curr_data[0])#.cuda()
    data = data.unsqueeze(0)
    data = data.float()
    data = data.to(device)
    return data

def getY(pkl_path, sample_path):
    curr_data = pkl.load(open(os.path.join(pkl_path, sample_path),'rb'))
    data = torch.from_numpy(curr_data[1])#.cuda()
    data = data.unsqueeze(0)
    data = data.float()
    data = data.to(device)
    return data

def get_item(pkl_path):
    return os.listdir(pkl_path)

def conv(ni, nf, ks=3, act=True):
    res = nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)
    if act: res = nn.Sequential(res, nn.ReLU())
    return res

def queue_loss(predictions, targes):
    m = nn.Softmax(dim=1)
    predictions = m(predictions)
    return ((predictions-targes)**2).mean()

def main():

    data_path = '/home/eliransc/projects/def-dkrass/eliransc/data/phasetype_data'

    dblock = DataBlock(get_items=get_item,
                       get_x=lambda x: getX(data_path, x),
                       get_y=lambda x: getY(data_path, x))

    dsets = dblock.datasets(data_path)
    dls = dblock.dataloaders(data_path, batch_size=256, num_workers=0)

    print(torch.cuda.is_available())

    simple_cnn = sequential(
        conv(1, 4),
        conv(4, 8),
        conv(8, 16),
        conv(16, 32),
        conv(32, 16),
        conv(16, 8),
        Flatten(),
        nn.Linear(32, 70),
        #     nn.ReLU(),
        #     nn.Linear(72,50),
        #     nn.ReLU(),
        #     nn.Linear(50,70),
    )

    learn = Learner(dls, simple_cnn, loss_func=queue_loss, metrics=queue_loss)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(learn.model.to(device))

    print(next(learn.model.parameters()).is_cuda )

    lr__ = learn.lr_find()

    print('finish')






if __name__ == "__main__":
    main()

