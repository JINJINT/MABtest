import numpy as np
from numpy.random import rand
np.set_printoptions(precision = 10)

import os
import logging, argparse
from datetime import datetime

from MAB import *
from MABprocess import *
from plotting import *
from toimport import *
from pathos import multiprocessing
from itertools import product
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def main():

    # Get arguments
    no_arms = args.no_arms
    delta = args.delta
    betalist = str2list(args.beta, 'float')

    mu_list = str2list(args.mu_list, 'float')
    if len(mu_list) != no_arms:
        print("The length of mu list does not equal to the number of arms, start draw randomly...")
        mu_list = rand(args.no_arms)

    print("The generated mean of the arms: ", ', '.join(map(str,mu_list)))
    
    print("The true best arm is %d" % (argmax(mu_list)))


    toplot = args.plot 
    toprint = args.print
    tosave = args.save

    direc = args.dir

    trials = args.trials

    def single(trial, beta):

        # initialize the bandit 
        mab_process = MABprocess(no_arms, 1, mu_list, delta = delta, beta = beta, verbose = toprint)
        # run the bandit (for now its just TTTS sampling with stopping at best arm)
        print('Running trial %d for beta %.2f'%(trial, beta))
        mab_process.run_MAB()
        print('Finished trial %d for beta %.2f'%(trial, beta))

        filename = 'MAB_TTTS_%.2fbeta_%.2fdelta_%.2fepsilon_BernoulliLIL_best1_%darms_trial%d' % (beta, delta, 0.0, no_arms, trial)
        
        if tosave:    
            mab_process.save_MAB(direc, filename)

        if trial == 1:
            mab_process.plot_MAB(direc, filename)

    def single_unpack(args):
        return single(*args)        


    with poolcontext(processes = 11) as p:
        p.map(single_unpack, product(list(range(trials)), betalist))
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-arms', type=int, default = 5)
    parser.add_argument('--mu-list', type=str, default = "0.1, 0.2, 0.3, 0.4, 0.5")
    parser.add_argument('--delta', type=float, default = 0.05)
    parser.add_argument('--beta', type=str, default = '0.5, 0.6, 0.7, 0.8, 0.9, 1')
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save', action='store_false')
    parser.add_argument('--dir', type = str, default = 'mabresults')
    parser.add_argument('--trials', type = int, default = 50)
    args = parser.parse_args()
    main()