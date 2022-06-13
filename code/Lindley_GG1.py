import numpy as np
import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
# import sympy
# from sympy import *
from scipy.special import gamma, factorial
from scipy.stats import gamma
import matplotlib.pyplot as plt
import os
import pandas as pd


def gamma_pdf(x, theta, k):
    return (1 / (gamma(k))) * (1 / theta ** k) * (np.exp(-x / theta))


def gamma_lst(s, theta, k):
    return (1 + theta * s) ** (-k)


def get_nth_moment(theta, k, n):
    s = Symbol('s')
    y = gamma_lst(s, theta, k)
    for i in range(n):
        if i == 0:
            dx = diff(y, s)
        else:
            dx = diff(dx, s)
    return ((-1) ** n) * dx.subs(s, 0)


def gamma_lst(s, theta, k):
    return (1 + theta * s) ** (-k)


def unif_lst(s, b, a=0):
    return (1 / (b - a)) * ((np.exp(-a * s) - np.exp(-b * s)) / s)


def n_mom_uniform(n, b, a=0):
    return (1 / ((n + 1) * (b - a))) * (b ** (n + 1) - a ** (n + 1))


def laplace_mgf(t, mu, b):
    return exp(mu * t) / (1 - (b ** 2) * (t ** 2))


def nthmomlap(mu, b, n):
    t = Symbol('t')
    y = laplace_mgf(t, mu, b)
    for i in range(n):
        if i == 0:
            dx = diff(y, t)
        else:
            dx = diff(dx, t)
    return dx.subs(t, 0)


def normal_mgf(t, mu, sig):
    return exp(mu * t + (sig ** 2) * (t ** 2) / 2)


def nthmomnormal(mu, sig, n):
    t = Symbol('t')
    y = normal_mgf(t, mu, sig)
    for i in range(n):
        if i == 0:
            dx = diff(y, t)
        else:
            dx = diff(dx, t)
    return dx.subs(t, 0)


def generate_unif(is_arrival):
    if is_arrival:
        b_arrive = np.random.uniform(2, 10)
        a_arrive = 0
        moms_arr = []
        for n in range(1, 11):
            moms_arr.append(n_mom_uniform(n, b_arrive))
        return (a_arrive, b_arrive, moms_arr)
    else:
        b_ser = 2
        a_ser = 0
        moms_ser = []
        for n in range(1, 11):
            moms_ser.append(n_mom_uniform(n, b_ser))
        return (a_ser, b_ser, moms_ser)


def generate_gamma(is_arrival):
    if is_arrival:
        rho = np.random.uniform(0.1, 0.99)
        shape = np.random.uniform(0.1, 100)
        scale = 1 / (rho * shape)
        moms_arr = np.array([])
        for mom in range(1, 11):
            moms_arr = np.append(moms_arr, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
        return (shape, scale, moms_arr)
    else:
        shape = np.random.uniform(1, 100)
        scale = 1 / shape
        moms_ser = np.array([])
        for mom in range(1, 11):
            moms_ser = np.append(moms_ser, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
        return (shape, scale, moms_ser)


def generate_normal(is_arrival):
    if is_arrival:
        mu = np.random.uniform(1.5, 10)
        sig = np.random.uniform(mu / 6, mu / 4)

        moms_arr = np.array([])
        for mom in tqdm(range(1, 11)):
            moms_arr = np.append(moms_arr, np.array(N(nthmomnormal(mu, sig, mom))).astype(np.float64))

        return (mu, sig, moms_arr)
    else:
        mu = 1
        sig = np.random.uniform(0.15, 0.22)

        moms_ser = np.array([])
        for mom in tqdm(range(1, 11)):
            moms_ser = np.append(moms_ser, np.array(N(nthmomnormal(mu, sig, mom))).astype(np.float64))
        return (mu, sig, moms_ser)

def main():

    pkl_path = '/home/eliransc/projects/def-dkrass/eliransc/Deep_queue/pkl/lindlyesgg1_1.pkl'

    # pkl_path = r'./lindlyesgg1.pkl'


    if os.path.exists(pkl_path):
        df = pkl.load(open(pkl_path, 'rb'))
    else:
        df = pd.DataFrame([], columns = [])

    b = 2.230380964364765
    num_trails = 250000000
    mu = 1
    arrivals = np.random.uniform(0, b, num_trails)  # np.random.exponential(2, num_trails)
    ser = np.random.exponential(mu, num_trails)

    from tqdm import tqdm
    waiting = []
    queueing = []
    for ind in range(num_trails):
        if ind == 0:
            queueing.append(0)
            waiting.append(ser[0])
        else:
            queueing.append(max(queueing[ind - 1] + ser[ind - 1] - arrivals[ind - 1], 0))
            waiting.append(queueing[ind] + ser[ind])

    curr_ind = df.shape[0]

    meanwait = np.array(waiting[5000:]).mean()
    mean_l = meanwait * (2 / b)

    df.loc[curr_ind, 'arrival_dist'] = 'uniform'
    df.loc[curr_ind, 'arrival_params'] = str(0)+'_' + str(b)
    df.loc[curr_ind, 'ser_dist'] = 'exp'
    df.loc[curr_ind, 'ser_params'] = str(mu)
    df.loc[curr_ind, 'avg_waiting'] = meanwait
    df.loc[curr_ind, 'avg_length'] = mean_l
    df.loc[curr_ind, 'num_iters'] = num_trails

    pkl.dump(df, open(pkl_path, 'wb'))

    print(df)




if __name__ == "__main__":

    main()