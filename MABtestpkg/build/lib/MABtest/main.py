import numpy as np
from numpy.random import rand
from numpy import argsort
import os
import logging, argparse
from datetime import datetime
from pathos import multiprocessing
from itertools import product
from contextlib import contextmanager
import warnings
import dill

from .runsingle import *

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def main():
    #warnings.filterwarnings('ignore')

    no_arms = args.no_arms
    topm = args.topm

    samp_type = args.samp_type
    reward_type = args.reward_type
    mu_type = args.mu_type
    timevar_type = args.timevar_type
    stop_type = args.stop_type
    
    conf_level = args.conf_level
    precision_level = args.precision_level
    
    record_break  = args.record_break
    record_max = args.record_max

    baseline = args.baseline

    trials = args.trials

    di = args.dir
    direc = '%s/%s/%s_%s_%d_topm%d_time%s/' %(di, reward_type, stop_type, mu_type, no_arms, topm, timevar_type) 
    auto_stop = args.auto_stop
    save = args.save
    
    time_detect = args.time_detect
    time_series = args.time_series
    real_methodlist = str2list(args.real_method, 'string')
    real_method = args.real_method

    # transform arguments
    trade_offlist = str2list(args.trade_off, 'float')
    time_decaylist = str2list(args.time_decay, 'float')
    record_time = lambda t: record_break*t
    toprint = args.toprint
    
    # read the mean list of arms / generate mean list of arms
    mu_list = str2list(args.mu_list, 'float')
    baseline = args.baseline
    
    mu_time_list = generate_mu(direc, record_max, no_arms, mu_type, timevar_type, reward_type, mu_list)
    baseline_time = lambda t: baseline


    def single(trial: int):
        import os
        from runsingle import runsingle
        from plotting import plot_MAB
        from numpy import argsort
        import dill       

        if trade_off == -1:
            samp_typ = 'uniform'
        else:
            samp_typ = samp_type  

        if 'Control' in stop_type:       
            filename = 'MAB_noarms%d_%s_%s_%s_%s_%d_trade_off%.2f_conf_level%.2f_precision_level%.2f_break%d_max%d_baseline%.2f_time_decay%.4f%s' %\
                (no_arms, samp_typ, reward_type, stop_type, timevar_type, topm, trade_off, conf_level, precision_level, record_break, record_max, baseline, time_decay, real_method)
        else:
            filename = 'MAB_noarms%d_%s_%s_%s_%s_%d_trade_off%.2f_conf_level%.2f_precision_level%.2f_break%d_max%d_time_decay%.4f%s' %\
                (no_arms, samp_typ, reward_type, stop_type, timevar_type, topm, trade_off, conf_level, precision_level, record_break, record_max, time_decay, real_method)
                    
        # Run experiment if data doesn't exist yet
        if not os.path.exists("%s/%s_trial%d.pkl"%(direc, filename, trial)):
            print('Running trial %d for trade_off %.2f ...'%(trial, trade_off))
            # initialize the bandit 
            mab_process = runsingle(mu_list: Callable[[int, int],float], 
                 no_arms: int, no_output: int, 
                 conf_level: float = 0.05, trade_off: float = 1, precision_level: float = 0.,
                 reward_type: str = 'Bernoulli', samp_type: str = 'TS', stop_type: str = 'Best', 
                 baseline: Callable[[int], float] = None, 
                 record_time : Callable[[int],float] = lambda t: 1000*t, record_max: int = 10**6, auto_stop : bool = False, 
                 is_timevar: bool = False, timevar_type: str= 'General', time_decay : float = 0.001, time_window: int = 1000,
                 verbose = False)
            
            # run the bandit
            mab_process.run()
            print('Whether reached stopping: %r' % mab_process.record_stop)
            print('Finished trial %d for trade_off %.2f !'%(trial, trade_off))
            if save:    
                mab_process.save(direc, filename+'_trial%d'%trial)
            
            if not os.path.exists(direc + '/plots'):
                os.makedirs(direc+'/plots')
            plot_filenames = [file for file in os.listdir(direc+'/plots') if file.startswith(filename)]    
            
            if plot_filenames==[]:
                plot_MAB(mab_process.mabdat, direc, filename+'_trial%d'%trial)            
        else:
            print('Trial %d for trade_off %.2f are already run.'%(trial, trade_off))
            if not os.path.exists(direc + '/plots'):
                os.makedirs(direc+'/plots')
            plot_filenames = [file for file in os.listdir(direc+'/plots') if file.startswith(filename)]    
            
            if plot_filenames==[]:
                with open("%s/%s_trial%d.pkl"%(direc, filename, trial), 'rb') as input:
                    mabdat = dill.load(input)
                plot_MAB(mabdat, direc, filename+'_trial%d'%trial)            
      
    def single_unpack(args):
        return single(*args)        

    with poolcontext(processes = 1) as p:
        p.map(single_unpack, range(trials))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    '''=========================================
           synthetic data related parameters
    ========================================='''

    # we offer several different type of true mean of reward to be generated automatically, for details of each type, see generatemu.py
    parser.add_argument('--mu-type', type=str, default = "biggap")
    
    # or we can manually specify the true mean of reward for each arm, based on which we simulate the data
    parser.add_argument('--mu-list', type=str, default = "0.5, 0.4, 0.3, 0.2, 0.1") 
    
    '''=========================================
           bandit algorithm related parameters
    ========================================='''
    # how many number of arms, should equal to the length of mu_list
    parser.add_argument('--no-arms', type=int, default = 5)

    # how many number of arms to output in the end, should smaller or equal to no_arms
    parser.add_argument('--no-output', type=int, default = 1)

    # the confidence level, usually set in [0.01, 0.1]
    parser.add_argument('--conf-level', type=float, default = 0.05)

    # the precision level in the comparison of true reward means among arms, can be positive or negative or zero, when set at non-zero, it changes 'mu_arm1 > mu_arm2' to 'mu_arm1 > mu_arm2 + precision_level'
    parser.add_argument('--precision-level', type=float, default = 0.0)

    # which base algorithm to choose
    parser.add_argument('--samp-type', type = str, default = 'UCB') # TS, UCB, uniform
    
    # if the base algorithm is not uniform, then we can additionally specify a trade-off parameter (in [1,\infty]) to baseline the extend of exploring sub-topmal arms, the higher it is, the more we will explore sub-optimal arms
    parser.add_argument('--trade-off', type = float, default = 1) 
    
    # what kind of stopping rule, i.e. what kind of testing that you want to reach significance
    parser.add_argument('--stop-type', type = 1, default = 'Best') # Best, Best-Base, FWER-Base, FDR-Base

    # if there is a part of baseline in the interested testing problem, then we can specify the baseline here
    parser.add_argument('--baseline', type = float, default = 0.) 

    # the type of the reward for each arm
    parser.add_argument('--reward-type', type = str, default = 'Bernoulli') # Bernoulli, Gaussian
    
    '''=========================================
           Time variaiton related parameters
    ========================================='''
    # whether there is time variation or not
    parser.add_argument('--is-timevar', action = 'store_true') 

    # if there is time variation, what types of time variation: 1. General; 2. Abruput; 3. (to come) Continuous.
    parser.add_argument('--timevar-type', type = str, default = 'General') 
    
    # if there is time variation, and the type of time variation is 'General', then we will use memory decay, and the decay rate can be specified here, usually use value in [0,0.1], and the higher, the faster it will decay
    parser.add_argument('--time-decay', type = float, default = 0.0) 

    # if there is time variation, and the type of time variation is '', then we will use memory decay, and the 
    parser.add_argument('--time-window', type = int, default = 1000) 
    
    # wheter to stop automatically according to testing criteria
    parser.add_argument('--auto-stop', action = 'store_true') 
    
    '''=========================================
           demonstration related parameters
    ========================================='''
    # over each how many pulls/samples, we record the stauts of bandits
    parser.add_argument('--record-break', type = int, default = 1000)
    
    # the maximum pulls/samples that we run the bandits
    parser.add_argument('--record-max', type = int, default = 10**6)
    
    # how many repitition/trials we run one bandits
    parser.add_argument('--trials', type = int, default = 1)
    
    # whether print out the results or not
    parser.add_argument('--toprint', action='store_true')
    
    # whether plot the results or not
    parser.add_argument('--plot', action='store_false')
    
    # whether save the results or not
    parser.add_argument('--save', action='store_false')
    
    # the directory to save all the plots/data for this experiments
    parser.add_argument('--dir', type = str, default = '/Users/jinjint/Desktop/MABcloud/')
    
    
    args = parser.parse_args()
    logging.info(args)
    main()






