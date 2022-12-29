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
from numpy.linalg import matrix_power

sys.path.append(r'C:\Users\user\workspace\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')


from butools.ph import *
from butools.map import *
from butools.queues import *
import time
from butools.mam import *
from butools.dph import *


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


# def generate_gamma(is_arrival):
#     if is_arrival:
#         rho = np.random.uniform(0.7, 0.99)
#         shape = np.random.uniform(0.1, 100)
#         scale = 1 / (rho * shape)
#         moms_arr = np.array([])
#         for mom in range(1, 11):
#             moms_arr = np.append(moms_arr, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
#         return (shape, scale, moms_arr)
#     else:
#         shape = np.random.uniform(1, 100)
#         scale = 1 / shape
#         moms_ser = np.array([])
#         for mom in range(1, 11):
#             moms_ser = np.append(moms_ser, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
#         return (shape, scale, moms_ser)

def get_hyper_ph_representation(mu):
    p1 = 0.5 + (-2. + 0.5 * mu) / (16. + -mu ** 2) ** 0.5
    p2 = 1 - p1
    lam1 = 1 / (2 + 0.5 * mu + 0.5 * (16 - mu ** 2) ** 0.5)
    lam2 = 1 / (2 + 0.5 * mu - 0.5 * (16 - mu ** 2) ** 0.5)

    s = np.array([[p1, p2]])
    A = np.array([[-lam1, 0], [0, -lam2]])
    return (s, A)

def log_normal_gener(mu, sig2, sample_size):
    m = np.log((mu**2)/(sig2+mu**2)**0.5)
    v = (np.log(sig2/mu**2+1))**0.5
    s = np.random.lognormal(m, v, sample_size)
    return s

def compute_first_ten_moms_log_N(s):
    moms = []
    for ind in range(1,11):
        moms.append((s**ind).mean())
    return moms

def generate_gamma(is_arrival, rho = 0.01):
    if is_arrival:
        # rho = np.random.uniform(0.7, 0.99)
        shape = 0.25/rho # 0.25 # np.random.uniform(0.1, 100)
        scale =  4 #1 / (rho * shape)
        moms_arr = np.array([])
        for mom in range(1, 11):
            moms_arr = np.append(moms_arr, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
        return (shape, scale, moms_arr)
    else:
        shape = 0.25 # np.random.uniform(1, 100)
        scale = 1 / shape
        moms_ser = np.array([])
        for mom in range(1, 11):
            moms_ser = np.append(moms_ser, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
        return (shape, scale, moms_ser)

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

def compute_first_n_moments(s, A, n=3):
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list




def create_Erlang4(lam):
    s = np.array([[1, 0, 0, 0]])

    A = np.array([[-lam, lam, 0, 0], [0, -lam, lam, 0], [0, 0, -lam, lam], [0, 0, 0, -lam]])

    return (s, A)


def generate_normal(is_arrival):
    if is_arrival:
        mu = np.random.uniform(1.1, 2.0)
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

    def __init__(self, inter_arrival, inter_service):


        self.num_cust_sys = 0
        self.num_cust_durations = np.zeros(1000)
        self.last_event_time = 0
        self.last_time = 0
        self.inter_arrival = inter_arrival
        self.inter_service = inter_service

    def service(self, env,  server, args, ind):


        with server[0].request() as req:
            yield req

            # service time


            yield env.timeout(self.inter_service[ind])

            tot_time = env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time
            self.num_cust_sys -= 1
            self.last_event_time = env.now
            self.last_time = env.now




    def customer_arrivals(self, env, server, args):

        ind = 0

        while ind < self.inter_arrival.shape[0]- 2:


            yield env.timeout(self.inter_arrival[ind])
            ind += 1

            tot_time = env.now - self.last_event_time
            self.num_cust_durations[self.num_cust_sys] += tot_time

            self.last_event_time = env.now
            self.num_cust_sys += 1
            self.last_time = env.now
            # print(env.now, self.num_cust_sys)
            env.process(self.service(env,  server, args, ind))


def gam(rho, m=1):
    return 0


def sig1(rho, m=1):
    return 1 + gam(rho)


def sig2(rho, m=1):
    return 1 - 4 * gam(rho)


def sig3(rho, m=1):
    return sig2(rho) * np.exp(-2 * (1 - rho) / (3 * rho))


def sig4(rho, m=1):
    return min(1, (sig1(rho) + sig3(rho)) / 2)


def psi(c2, rho, m=1):
    if c2 >= 1:
        return 1
    else:
        return (sig4(rho)) ** (2 * (1 - c2))


def EWmm1(rho, lambd):
    return rho / ((1 - rho) * lambd)


def phi(rho, ca2, cs2, m=1):
    if ca2 >= cs2:
        part1 = ((4 * (ca2 - cs2)) / (4 * ca2 - 3 * cs2)) * sig1(rho)
        part2 = ((cs2) / (4 * ca2 - 3 * cs2)) * psi((ca2 + cs2) / 2, rho)

        return part1 + part2
    else:
        part1 = (((cs2 - ca2)) / (2 * ca2 + 2 * cs2)) * sig3(rho)
        part2 = ((cs2 + 3 * ca2) / (2 * ca2 + 2 * cs2)) * psi((ca2 + cs2) / 2, rho)

        return part1 + part2


def EW93(rho, ca2, cs2, lambd, m=1):
    return phi(rho, ca2, cs2) * ((ca2 + cs2) / 2) * EWmm1(rho, lambd)


def g(rho, ca2, cs2):
    if ca2 < 1:
        return np.exp(-((2 * (1 - rho) / (3 * rho))) * (((1 - ca2) ** 2)) * (ca2 + cs2))
    else:
        return 1


def EW83(mean_s, rho, ca2, cs2):
    return mean_s * rho * (ca2 + cs2) * g(rho, ca2, cs2) / ((2 * (1 - rho)))


def main(args):


    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    arrival_dics = {0: 'LN025', 1: 'LN4', 2: 'G4', 3: 'E4', 4: 'H2'}

    ser_dics = {0: 'LN025', 1: 'LN4', 2: 'G4', 3: 'E4', 4: 'H2', 5: 'M'}

    sample_size = args.sample_size

    import time
    cur_time = int(time.time())

    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed)
    print(seed)
    rhos = np.linspace(0.01, 0.96, 20)

    ind = np.random.randint(0, 20)

    rho = rhos[ind]

    rho = 0.76
    print(rho)

    df_sum = pd.DataFrame([])

    for arrival_dist in arrival_dics.keys():

        for service_dist in tqdm(ser_dics.keys()):

            ## Arrival dist and realizations
            if arrival_dics[arrival_dist] == 'LN025':
                mu = 1/rho
                sig2 = 0.25*mu
                inter_arrival = log_normal_gener(mu, sig2, sample_size)
                m = np.log((mu ** 2) / (sig2 + mu ** 2) ** 0.5)
                v = (np.log(sig2 / mu ** 2 + 1)) ** 0.5
                moms_arrive =  compute_first_ten_moms_log_N(inter_arrival)
                arrival_csv = 0.25

            elif arrival_dics[arrival_dist] == 'LN4':

                mu = 1 / rho
                sig2 = 4 * mu
                inter_arrival = log_normal_gener(mu, sig2, sample_size)
                m = np.log((mu ** 2) / (sig2 + mu ** 2) ** 0.5)
                v = (np.log(sig2 / mu ** 2 + 1)) ** 0.5
                moms_arrive = compute_first_ten_moms_log_N(inter_arrival)
                arrival_csv = 4

            elif arrival_dics[arrival_dist] == 'G4':

                shape, scale, moms_arrive = generate_gamma(True, rho)
                moms_arrive = compute_first_ten_moms_log_N(inter_arrival)
                inter_arrival = np.random.gamma(shape, scale, sample_size)
                moms_arrive = compute_first_ten_moms_log_N(inter_arrival)
                arrival_csv = 4

            elif arrival_dics[arrival_dist] == 'E4':

                lam = 4*rho
                s, A = create_Erlang4(lam)
                inter_arrival = SamplesFromPH(ml.matrix(s), A, sample_size)
                # moms_arrive = compute_first_n_moments(s, A, 10)
                # moms_arrive = np.array(moms_arrive).flatten()
                moms_arrive = compute_first_ten_moms_log_N(inter_arrival)
                arrival_csv = 1/lam

            elif arrival_dics[arrival_dist] == 'H2':

                mu = 1/rho
                s, A = get_hyper_ph_representation(mu)
                inter_arrival = SamplesFromPH(ml.matrix(s), A, sample_size)
                # moms_arrive = compute_first_n_moments(s, A, 10)
                # moms_arrive = np.array(moms_arrive).flatten()
                moms_arrive = compute_first_ten_moms_log_N(inter_arrival)
                arrival_csv = 4



            #####################################
            ## Service dist and realizations
            #####################################

            if ser_dics[service_dist] == 'LN025':
                mu = 1
                sig2 = 0.25*mu
                inter_service = log_normal_gener(mu, sig2, sample_size)
                m = np.log((mu ** 2) / (sig2 + mu ** 2) ** 0.5)
                v = (np.log(sig2 / mu ** 2 + 1)) ** 0.5
                moms_service = compute_first_ten_moms_log_N(inter_service)
                service_csv = 0.25

            elif ser_dics[service_dist] == 'LN4':

                mu = 1
                sig2 = 4 * mu
                inter_service = log_normal_gener(mu, sig2, sample_size)
                m = np.log((mu ** 2) / (sig2 + mu ** 2) ** 0.5)
                v = (np.log(sig2 / mu ** 2 + 1)) ** 0.5
                moms_service = compute_first_ten_moms_log_N(inter_service)
                service_csv = 4

            elif ser_dics[service_dist] == 'G4':

                shape, scale, moms_service = generate_gamma(False)
                inter_service = np.random.gamma(shape, scale, sample_size)
                moms_service = compute_first_ten_moms_log_N(inter_service)
                service_csv = 4

            elif ser_dics[service_dist] == 'E4':

                lam = 4
                s, A = create_Erlang4(lam)
                inter_service = SamplesFromPH(ml.matrix(s), A, sample_size)
                # moms_service = compute_first_n_moments(s, A, 10)
                # moms_service = np.array(moms_service).flatten()
                moms_service = compute_first_ten_moms_log_N(inter_service)
                service_csv = 1/lam

            elif ser_dics[service_dist] == 'H2':

                mu = 1
                s, A = get_hyper_ph_representation(mu)
                inter_service = SamplesFromPH(ml.matrix(s), A, sample_size)
                # moms_service = compute_first_n_moments(s, A, 10)
                # moms_service = np.array(moms_service).flatten()
                moms_service = compute_first_ten_moms_log_N(inter_service)
                service_csv = 4

            elif ser_dics[service_dist] == 'M':

                mu = 1
                inter_service = np.random.exponential(1, sample_size)
                moms_service = compute_first_ten_moms_log_N(inter_service)
                service_csv = 1


            curr_ind = df_sum.shape[0]
            df_sum.loc[curr_ind, 'arrival_dist'] = arrival_dics[arrival_dist]
            df_sum.loc[curr_ind, 'service_dist'] = ser_dics[service_dist]
            df_sum.loc[curr_ind, 'rho'] = rho
            df_sum.loc[curr_ind, 'arrival_rate'] = rho


            for mom in range(1,11):

                df_sum.loc[curr_ind, 'mom'+str(mom)+'arrival'] = moms_arrive[mom-1]
                df_sum.loc[curr_ind, 'mom' + str(mom) + 'service'] = moms_service[mom-1]

            df_sum.loc[curr_ind, 'Whitt83'] = EW83(moms_service[0], rho, arrival_csv, service_csv)
            df_sum.loc[curr_ind, 'Whitt93'] = EW93(rho, arrival_csv, service_csv, rho)


            gg1 = GG1(inter_arrival, inter_service)

            start_time = time.time()

            env = simpy.Environment()

            server = []
            for server_ind in range(args.size):
                server.append(simpy.Resource(env, capacity=1))


            args.end_time = float(500000000 / rho)


            env.process(gg1.customer_arrivals(env, server, args))
            env.run(until=(args.end_time))

            steady_state = gg1.num_cust_durations / gg1.num_cust_durations.sum()

            mean_L = (np.arange(1000)*steady_state).sum()
            print('mean L: ', mean_L)

            df_sum.loc[curr_ind, 'sim_L'] = mean_L
            df_sum.loc[curr_ind, 'sim_W'] = mean_L/rho

            for ind in range(1000):
                df_sum.loc[curr_ind, 'steady' + str(ind)] = steady_state[ind]

            df_sum_full_path = args.df_summ + '_rho_' + str(rho) + '.pkl'

            pkl.dump(df_sum, open(df_sum_full_path, 'wb'))



def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, help='external arrivals', default=5000000)
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--rho', type=float, help='the prob of external matched customer', default=0.01)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([]))
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=1000)
    parser.add_argument('--size', type=int, help='the number of stations in the system', default=1)
    parser.add_argument('--p_correct', type=float, help='the prob of external matched customer', default=0.5)
    parser.add_argument('--ser_matched_rate', type=float, help='service rate of matched customers', default=1.2)
    parser.add_argument('--ser_mis_matched_rate', type=float, help='service rate of mismatched customers', default=10.)
    parser.add_argument('--num_iterations', type=float, help='service rate of mismatched customers', default=500)
    parser.add_argument('--case_num', type=int, help='case number in my settings', default=random.randint(0, 100000))
    parser.add_argument('--df_summ', type=str, help='case number in my settings', default='../pkl2/df_sum_res_sim_gg1_14')
    parser.add_argument('--is_corr', type=bool, help='should we keep track on inter departure', default=True)
    parser.add_argument('--waiting_pkl_path', type=bool, help='the path of the average waiting time', default='../pkl/waiting_time')

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

