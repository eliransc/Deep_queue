# imports
import simpy
import sys
sys.path.append(r'G:\My Drive\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')

import numpy as np

import argparse
import pandas as pd
import pickle as pkl
import time
import os
from tqdm import tqdm
from datetime import datetime
import random
from sampling_ph_1 import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'G:\My Drive\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')

import os
import pandas as pd
import argparse
from tqdm import tqdm
from butools.ph import *
from butools.map import *
from butools.queues import *
import time
from butools.mam import *
from butools.dph import *
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
import random
from scipy.stats import loguniform
from butools.fitting import *
from datetime import datetime
from fastbook import *
import itertools
from scipy.special import factorial
import pickle as pkl




def main(args):

    if sys.platform == 'linux':

        pass
    else:
        pass

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    random.seed(current_time)

    df_summary_result = pd.DataFrame([])
    for ind in tqdm(range(args.num_iterations)):

        start_time = time.time()

        env = simpy.Environment()

        server = []
        for server_ind in range(args.size):
            server.append(simpy.Resource(env, capacity=1))

        sums = [0, 0, 0, 0, 0, 0, 0, 0]

        corr_time = [0]
        corr_path = '../pkl/corr_time' + str(args.case_num) + '.pkl'
        pkl.dump(corr_time, open(corr_path, 'wb'))

        cur_time = int(time.time())
        np.random.seed(cur_time)

        arrival_rate = np.random.uniform(0.3, 0.999)

        max_ph_size = 100

        # service1 = send_to_the_right_generator(np.random.randint(1,3), max_ph_size)

        aa = np.array([0.5,0.5])
        AA = np.array([[-3,0],[0,-0.6]])
        AA = np.array([[-1,0],[0,-1]])

        service1 = (aa,AA)

        moms1 = compute_first_n_moments(service1[0], service1[1],10)

        service2 = send_to_the_right_generator(np.random.randint(1,3), max_ph_size)

        rho_2 = np.random.uniform(0, 0.999)
        service2 = (service2[0], service2[1]/(rho_2/arrival_rate))
        moms2 = compute_first_n_moments(service2[0], service2[1], 10)


        var1 = moms1[1][0]-moms1[0][0]**2
        L = arrival_rate+(arrival_rate**2+(arrival_rate**2) * var1)/(2*(1-arrival_rate))
        W = L/arrival_rate
        print('Averager waiting time in station 0 is: ', W)
        services = (service1, service2)

        avg_time = list(np.zeros(args.size))

        print('Rho_1: ', arrival_rate)
        print('Rho_2: ', rho_2)

        env.process(customer_arrivals(env, args.size, server, arrival_rate, services, args.case_num, avg_time, sums, args))
        env.run(until=(args.end_time))


        # total_avg_system = 0
        # for station_ind in range(args.size):
        #     df_summary_result.loc[ind, 'Arrival_'+str(station_ind)] = str(args.r[station_ind])
        #     df_summary_result.loc[ind, 'Service_rate' + str(station_ind)] = str(args.mu[station_ind])
        #     df_summary_result.loc[ind, 'avg_waiting_'+str(station_ind)] = avg_waiting[station_ind]
        #     df_summary_result.loc[ind, 'avg_sys_'+str(station_ind)] = avg_waiting[station_ind] *\
        #                                                               (np.sum(args.r[station_ind, :]) +
        #                                                                np.sum(args.r[:, station_ind])
        #                                                                -args.r[station_ind, station_ind])
        #     total_avg_system += df_summary_result.loc[ind, 'avg_sys_'+str(station_ind)]
        #     if station_ind == 0:
        #         df_summary_result.loc[ind, 'avg_sys_mg1_' + str(station_ind)], rho = avg_sys_station_0(args.r, args.mu,
        #                                                                                      station_ind)
        #         df_summary_result.loc[ind, 'avg_sys_mm1_' + str(station_ind)] = rho / (1 - rho)
        #
        #     else:
        #         df_summary_result.loc[ind, 'avg_sys_mg1_'+str(station_ind)], rho = avg_sys(args.r, args.mu, station_ind)
        #         df_summary_result.loc[ind, 'avg_sys_mm1_' + str(station_ind)] = rho/(1-rho)
        #
        #
        # df_summary_result.loc[ind, 'avg_sys_total'] = total_avg_system
        # print(df_summary_result)
        #
        # print("--- %s seconds the %d th iteration ---" % (time.time() - start_time, 1))
        #
        # with open('../pkl/df_summary_result_sim_different_sizes_queues_'+str(current_time)+'.pkl', 'wb') as f:
        #     pkl.dump(df_summary_result, f)
        # print('The average number of customers in station 1 is: ', df_summary_result.loc[0,'avg_sys_1'])
        #
        # print('The average is station 0 is: ', avg_waiting[0] * (lam00+lam01))
        # print('The average is station 1 is: ', df_summary_result.loc[0, 'avg_sys_1'])
        #
        #
        # if not os.path.exists(args.df_summ):
        #     df_ = pd.DataFrame([])
        # else:
        #     df_ = pkl.load(open(args.df_summ, 'rb'))
        # ind = df_.shape[0]
        #
        # df_.loc[ind, 'lam00'] = lam00
        # df_.loc[ind, 'lam01'] = lam01
        # df_.loc[ind, 'lam10'] = lam10
        # df_.loc[ind, 'lam11'] = lam11
        #
        # df_.loc[ind, 'mu00'] = mu00
        # df_.loc[ind, 'mu01'] = mu01
        # df_.loc[ind, 'mu10'] =  0 #mu10
        # df_.loc[ind, 'mu11'] = mu11
        #
        # df_.loc[ind, 'avg_cust_0'] = avg_waiting[0] * (lam00+lam01)
        # df_.loc[ind, 'avg_cust_1'] = df_summary_result.loc[0, 'avg_sys_1']
        # df_.loc[ind, 'avg_wait_0'] = avg_waiting[0]
        # df_.loc[ind, 'avg_wait_1'] = avg_waiting[1]
        # # df.loc[ind,'var_0'] = df_inter_departure_station_0['inter_departure_time'].var()
        #
        # df_.loc[ind, 'ind'] = case_ind
        # corr_time = pkl.load(open(corr_path, 'rb'))
        # df_.loc[ind, 'inter_rho'] = corr_time[-1]
        #
        # pkl.dump(df_, open(args.df_summ,'wb'))
        #
        # print(df_)

def service(env,  name, server, arrival_time, services, station,  case_num, avg_time, sums, args):

    if (np.remainder(name[station], 1000) == 0) & (station == 0):
        print('The current time is: ', env.now)
        print(avg_time)

        if sums[7] > 10:
            corr_path = '../pkl/corr_time' + str(args.case_num) + '.pkl'
            corr_time = pkl.load( open(corr_path, 'rb'))
            curr_corr = (sums[7]*sums[6]-sums[2]*sums[4])/(((sums[7]*sums[3]-sums[2]**2)**0.5)*((sums[7]*sums[5]-sums[4]**2)**0.5))
            print('The inter-departure correlation', curr_corr)
            corr_time.append(curr_corr)
            pkl.dump(corr_time, open(corr_path, 'wb'))

        # with open('../pkl/avg_waiting'+str(args.case_num), 'rb') as f:
        #     avg_waiting = pkl.load(f)
        # print('The average sys in station 1 is: ',avg_time[station_ind] *(np.sum(args.r[station_ind, :]) +
        #                                                                np.sum(args.r[:, station_ind])
        #                                                                -args.r[station_ind, station_ind]))

        # if sums[7] > 10:
        #     corr_path = '../pkl/corr_time' + str(args.case_num) + '.pkl'
        #     corr_time = pkl.load( open(corr_path, 'rb'))
        #     curr_corr = (sums[7]*sums[6]-sums[2]*sums[4])/(((sums[7]*sums[3]-sums[2]**2)**0.5)*((sums[7]*sums[5]-sums[4]**2)**0.5))
        #     print(curr_corr)
        #     corr_time.append(curr_corr)
        #     pkl.dump(corr_time, open(corr_path, 'wb'))
        #
        #     waiting_path = '../pkl/waiting_time' + str(args.case_num) + '.pkl'
        #     curr_waiting = pkl.load(open(waiting_path, 'rb'))
        #     curr_waiting[0] = avg_time[0]
        #     curr_waiting[1] = avg_time[1]
        #     pkl.dump(curr_waiting, open(waiting_path, 'wb'))

    with server[station].request() as req:
        yield req

        # service time

        s1 = services[station][0]
        A1 = services[station][1]

        s1 = s1.reshape((1, s1.shape[0]))
        ser_time = SamplesFromPH(ml.matrix(s1), A1, 1)

        yield env.timeout(ser_time)

        waiting_time = env.now - arrival_time

        name[station] += 1
        curr_waiting = (avg_time[station] * name[station]) / (name[station] + 1) + waiting_time / (name[station] + 1)

        avg_time[station] = curr_waiting



        # if customer is mismatched then she is redirected to the her designated queue
        if station == 0:

                station  = 1
                arrival_time = env.now

                if args.is_corr:


                    if (sums[0] == 0) & (sums[1] == 0):
                        sums[0] = arrival_time
                    else:
                        if sums[1] > 0:  # cur_ind > 0:
                            prev = sums[1]
                            curr = arrival_time - sums[0]
                            sums[0] = arrival_time
                            sums[1] = curr
                            sums[2] += prev
                            sums[3] += prev**2
                            sums[4] += curr
                            sums[5] += curr**2
                            sums[6] += prev*curr
                            sums[7] += 1

                        else:
                            sums[1] = arrival_time - sums[0]
                            sums[0] = arrival_time



                env.process(service(env, name, server, arrival_time, services, station,  case_num, avg_time, sums, args))


def customer_arrivals(env, size,  server, arrival_rate, services,  case_num, avg_time, sums, args):

    name = np.ones(size) * (-1)

    while True:

        # get the external stream identity

        yield env.timeout(np.random.exponential(1 / arrival_rate))

        arrival_time = env.now

        station = 0
        env.process(service(env, name, server, arrival_time, services, station,  case_num, avg_time, sums, args))


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([]))
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([]))
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=10000000)
    parser.add_argument('--size', type=int, help='the number of stations in the system', default=2)
    parser.add_argument('--p_correct', type=float, help='the prob of external matched customer', default=0.5)
    parser.add_argument('--ser_matched_rate', type=float, help='service rate of matched customers', default=1.2)
    parser.add_argument('--ser_mis_matched_rate', type=float, help='service rate of mismatched customers', default=10.)
    parser.add_argument('--num_iterations', type=float, help='service rate of mismatched customers', default=1)
    parser.add_argument('--case_num', type=int, help='case number in my settings', default=random.randint(0, 100000))
    parser.add_argument('--df_summ', type=str, help='case number in my settings', default='../pkl/df_sum_res_sim_32.pkl')
    parser.add_argument('--is_corr', type=bool, help='should we keep track on inter departure', default=True)

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

