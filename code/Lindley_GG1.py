import numpy as np
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt

from scipy.special import gamma, factorial
from scipy.stats import gamma
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import sys
import sympy
from sympy import *
import time
import datetime

def gamma_pdf(x, theta, k):
    return (1 / (gamma(k))) * (1 / theta ** k) * (np.exp(-x / theta))


def gamma_lst(s, theta, k):
    return (1 + theta * s) ** (-k)


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
        b_arrive = np.random.uniform(2, 8)
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
        rho = np.random.uniform(0.3, 0.99)
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
        mu = np.random.uniform(1.5, 7.5)
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

def main(args):


    now = datetime.datetime.now()
    current_time = now.strftime("%H_%M_%S")
    args.df_summ = args.df_summ + '_' + current_time + '_' + str(np.random.randint(1, 100000)) + '.pkl'

    for exmaples in range(200):


        arrival_dist = np.random.choice([1, 2, 3], size=1, replace=True, p=[0.3, 0.4, 0.3])[0]
        ser_dist = np.random.choice([1, 2, 3], size=1, replace=True, p=[0.3, 0.4, 0.3])[0]

        if arrival_dist == 1:
            arrival_dist_params = generate_unif(True)
        elif arrival_dist == 2:
            arrival_dist_params = generate_gamma(True)
        else:
            arrival_dist_params = generate_normal(True)

        if ser_dist == 1:
            ser_dist_params = generate_unif(False)
        elif ser_dist == 2:
            ser_dist_params = generate_gamma(False)
        else:
            ser_dist_params = generate_normal(False)

        print(arrival_dist_params, ser_dist_params, arrival_dist, ser_dist)

        arrival_rate = 1 / arrival_dist_params[2][0]

        mean_l_list = []
        mean_waiting_list = []

        for ind_iter in range(args.num_iterations):

            num_trails = args.num_trails


            if arrival_dist == 1: # if arrival is uniform
                a = arrival_dist_params[0]
                b = arrival_dist_params[1]
                arrivals = np.random.uniform(a, b, num_trails)
            elif arrival_dist == 2: # if arrival is gamma
                shape = arrival_dist_params[0]
                scale = arrival_dist_params[1]
                arrivals = np.random.gamma(shape, scale, num_trails)
            else:  # if arrival is normal
                mu = arrival_dist_params[0]
                sig = arrival_dist_params[1]
                arrivals = np.random.normal(mu, sig, num_trails)
                arrivals = np.where(arrivals < 0, 0, arrivals)  # if we get negative vals
                print('Number of negatives: ', arrivals[arrivals == 0].shape[0])


            if ser_dist == 1:  # if service is unifrom
                a_ser = ser_dist_params[0]
                b_ser = ser_dist_params[1]
                ser = np.random.uniform(a_ser, b_ser, num_trails)
            elif ser_dist == 2:  # if service is gamma
                shape_ser = ser_dist_params[0]
                scale_ser = ser_dist_params[1]
                ser = np.random.gamma(shape_ser, scale_ser, num_trails)
            else: # if service is normal
                mu_ser = ser_dist_params[0]
                sig_ser = ser_dist_params[1]
                ser = np.random.normal(mu_ser, sig_ser, num_trails)
                ser = np.where(ser < 0, 0, ser)  # if we get negative vals
                print('Number of negatives: ', arrivals[arrivals == 0].shape[0])




            waiting = []
            queueing = []
            for ind in range(num_trails):
                if ind == 0:
                    queueing.append(0)
                    waiting.append(ser[0])
                else:
                    queueing.append(max(queueing[ind - 1] + ser[ind - 1] - arrivals[ind - 1], 0))
                    waiting.append(queueing[ind] + ser[ind])


            meanwait = np.array(waiting[5000:]).mean()
            mean_waiting_list.append(meanwait)
            mean_l = meanwait * arrival_rate
            mean_l_list.append(mean_l)

            rho = arrival_rate*ser_dist_params[2][0]


        if not os.path.exists(args.df_summ):
            df_ = pd.DataFrame([])
        else:
            df_ = pkl.load(open(args.df_summ, 'rb'))
        ind = df_.shape[0]

        if arrival_dist == 1:
            df_.loc[ind, 'arrival_dist'] = 'Uniform'
            df_.loc[ind, 'arrival_params'] = str(arrival_dist_params[0]) + '_' + str(arrival_dist_params[1])

        elif arrival_dist == 2:
            df_.loc[ind, 'arrival_dist'] = 'Gamma'
            df_.loc[ind, 'arrival_params'] = str(arrival_dist_params[0]) + '_' + str(arrival_dist_params[1])
        else:
            df_.loc[ind, 'arrival_dist'] = 'Normal'
            df_.loc[ind, 'arrival_params'] = str(arrival_dist_params[0]) + '_' + str(arrival_dist_params[1])

        if ser_dist == 1:
            df_.loc[ind, 'ser_dist'] = 'Uniform'
            df_.loc[ind, 'ser_params'] = str(ser_dist_params[0]) + '_' + str(ser_dist_params[1])
        elif ser_dist == 2:
            df_.loc[ind, 'ser_dist'] = 'Gamma'
            df_.loc[ind, 'ser_params'] = str(ser_dist_params[0]) + '_' + str(ser_dist_params[1])
        else:
            df_.loc[ind, 'ser_dist'] = 'Normal'
            df_.loc[ind, 'ser_params'] = str(ser_dist_params[0]) + '_' + str(ser_dist_params[1])

        df_.loc[ind, 'arrival_expected'] = arrival_dist_params[2][0]
        df_.loc[ind, 'ser_expected'] = ser_dist_params[2][0]

        df_.loc[ind, 'avg_cust'] = np.array(mean_l_list).mean()
        df_.loc[ind, 'avg_wait'] = np.array(mean_waiting_list).mean()
        df_.loc[ind, 'arrival rate'] = arrival_rate
        df_.loc[ind, 'sim_runtime'] = args.num_trails
        df_.loc[ind, 'rho'] = rho
        df_.loc[ind, 'num_trails'] = args.num_iterations

        for mom in range(1, 1+np.array(arrival_dist_params[2]).shape[0]):
            df_.loc[ind, 'arrive_moms_'+str(mom)] = arrival_dist_params[2][mom-1]
            df_.loc[ind, 'ser_moms_'+str(mom)] = ser_dist_params[2][mom-1]


        pkl.dump(df_, open(args.df_summ, 'wb'))

        print(df_)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trails', type=int, help='The end of the simulation', default=200000000)
    parser.add_argument('--size', type=int, help='the number of stations in the system', default=1)
    parser.add_argument('--num_iterations', type=float, help='service rate of mismatched customers', default=2)
    parser.add_argument('--df_summ', type=str, help='case number in my settings', default='../pkl_1/df_sum_res_sim_gg1_Lindley')
    args = parser.parse_args(argv)

    return args

if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    main(args)