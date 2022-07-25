# imports
import simpy
import numpy as np
import sys
import argparse
import pandas as pd
import pickle as pkl
import time
import os
from tqdm import tqdm
from datetime import datetime
import random
import pickle as pkl
import matplotlib.pyplot as plt
import sympy
from sympy import *
from scipy.special import gamma, factorial


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


# def get_nth_moment(theta, k, n):
#     s = Symbol('s')
#     y = gamma_lst(s, theta, k)
#     for i in range(n):
#         if i == 0:
#             dx = diff(y, s)
#         else:
#             dx = diff(dx, s)
#     return ((-1) ** n) * dx.subs(s, 0)


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
        b_arrive = np.random.uniform(2, 4)
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
        rho = np.random.uniform(0.5, 0.99)
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
        mu = np.random.uniform(1.5, 3)
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



class GG1:

    def __init__(self, arrival_dist, ser_dist, arrival_dist_params, ser_dist_params):

        self.arrival_dist = arrival_dist
        self.ser_dist = ser_dist
        self.arrival_dist_params = arrival_dist_params
        self.ser_dist_params = ser_dist_params
        self.arrival_rate = 1/arrival_dist_params[2][0]
        self.num_cust_sys = 0
        self.num_cust_durations = np.zeros(500)
        self.last_event_time = 0
        self.last_time = 0

    def service(self, env,  server, args):

        station_ind = 0

        with server[0].request() as req:
            yield req

            # service time

            if self.ser_dist == 1:
                a = self.ser_dist_params[0]
                b = self.ser_dist_params[1]
                inter_ser = np.random.uniform(a, b)

                yield env.timeout(inter_ser)


            elif self.ser_dist == 2:
                shape = self.ser_dist_params[0]
                scale = self.ser_dist_params[1]
                inter_ser = np.random.gamma(shape, scale)
                yield env.timeout(inter_ser)
            else:
                mu = self.ser_dist_params[0]
                sig = self.ser_dist_params[1]
                inter_ser = np.max([0.000001, np.random.normal(mu, sig)])
                yield env.timeout(inter_ser)

            tot_time = env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time
            self.num_cust_sys -= 1
            self.last_event_time = env.now
            self.last_time = env.now

            # print(env.now, self.num_cust_sys)



    def customer_arrivals(self, env, server, args):

        while True:

            if self.arrival_dist == 1:
                a_arrive = self.arrival_dist_params[0]
                b_arrive = self.arrival_dist_params[1]
                inter_arrival = np.random.uniform(a_arrive, b_arrive)
                yield env.timeout(inter_arrival)

            elif self.arrival_dist == 2:
                shape = self.arrival_dist_params[0]
                scale = self.arrival_dist_params[1]
                inter_arrival = np.random.gamma(shape, scale)
                yield env.timeout(inter_arrival)
            else:
                mu = self.arrival_dist_params[0]
                sig = self.arrival_dist_params[1]
                inter_arrival = np.max([0.000001, np.random.normal(mu, sig)])
                yield env.timeout(inter_arrival)

            tot_time = env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time
            # if self.num_cust_sys == 0:
            #     print(self.num_cust_durations[self.num_cust_sys]/env.now, 1-self.arrival_rate*self.ser_dist_params[2][0], env.now)

            self.last_event_time = env.now
            self.num_cust_sys += 1
            self.last_time = env.now
            # print(env.now, self.num_cust_sys)
            env.process(self.service(env,  server, args))


def main(args):



    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    for ind in range(args.num_iterations):

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

        dist_dict = {1: 'uniform', 2: 'gamma', 3: 'normal'}

        # print(arrival_dist_params, ser_dist_params, arrival_dist, ser_dist)

        arrival_rate = 1 / arrival_dist_params[2][0]

        print('rho is: ', arrival_rate * ser_dist_params[2][0])
        print('arrival dist: ', dist_dict[arrival_dist])
        print('service dist: ', dist_dict[ser_dist])



        gg1 = GG1(arrival_dist, ser_dist, arrival_dist_params, ser_dist_params)

        start_time = time.time()

        env = simpy.Environment()

        server = []
        for server_ind in range(args.size):
            server.append(simpy.Resource(env, capacity=1))


        args.end_time = float(100000 / arrival_rate)


        env.process(gg1.customer_arrivals(env, server, args))
        env.run(until=(args.end_time))

        steady_state = gg1.num_cust_durations / args.end_time

        print(gg1.num_cust_durations[0]/args.end_time, 1-gg1.arrival_rate*gg1.ser_dist_params[2][0], args.end_time)
        print(gg1.num_cust_durations[0] / gg1.last_time,
              1 - gg1.arrival_rate * gg1.ser_dist_params[2][0], args.end_time)

        print((gg1.num_cust_durations / gg1.last_time)[:20])
        print(np.sum(gg1.num_cust_durations / gg1.last_time))

        print("--- %s seconds the %d th iteration ---" % (time.time() - start_time, ind))

        if os.path.exists(args.df_summ):
            df_summary_result = pkl.load(open(args.df_summ, 'rb'))
        else:
            df_summary_result = pd.DataFrame([])


        curr_ind = df_summary_result.shape[0]

        df_summary_result.loc[curr_ind, 'arrive_dist'] = dist_dict[arrival_dist]
        df_summary_result.loc[curr_ind, 'ser_dist'] = dist_dict[ser_dist]
        df_summary_result.loc[curr_ind, 'arrive_dist_params_1'] = arrival_dist_params[0]
        df_summary_result.loc[curr_ind, 'arrive_dist_params_2'] = arrival_dist_params[1]
        df_summary_result.loc[curr_ind, 'ser_dist_params_1'] = ser_dist_params[0]
        df_summary_result.loc[curr_ind, 'ser_dist_params_2'] = ser_dist_params[1]
        for df_ind in range(1, 11):
            df_summary_result.loc[curr_ind, 'arrive_dist_moms'+str(df_ind)] = arrival_dist_params[2][df_ind-1]
            df_summary_result.loc[curr_ind, 'ser_dist_moms' + str(df_ind)] = ser_dist_params[2][df_ind - 1]

        for df_ind in range(100):
            df_summary_result.loc[curr_ind, 'steady_probs'+str(df_ind)] = steady_state[df_ind]

        df_summary_result.loc[curr_ind, 'sum_100'] = np.sum(steady_state[:100])

        pkl.dump(df_summary_result, open(args.df_summ, 'wb'))


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([]))
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([]))
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=1000)
    parser.add_argument('--size', type=int, help='the number of stations in the system', default=1)
    parser.add_argument('--p_correct', type=float, help='the prob of external matched customer', default=0.5)
    parser.add_argument('--ser_matched_rate', type=float, help='service rate of matched customers', default=1.2)
    parser.add_argument('--ser_mis_matched_rate', type=float, help='service rate of mismatched customers', default=10.)
    parser.add_argument('--num_iterations', type=float, help='service rate of mismatched customers', default=5)
    parser.add_argument('--case_num', type=int, help='case number in my settings', default=random.randint(0, 100000))
    parser.add_argument('--df_summ', type=str, help='case number in my settings', default='../pkl/df_sum_res_sim_gg1_1.pkl')
    parser.add_argument('--is_corr', type=bool, help='should we keep track on inter departure', default=True)
    parser.add_argument('--waiting_pkl_path', type=bool, help='the path of the average waiting time', default='../pkl/waiting_time')

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

