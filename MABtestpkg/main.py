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
from pathos.multiprocessing import ProcessingPool as Pool

from MABtest.runsingle import *
from MABtest.toimport import *

from MABtest.plotting import *
from MABtest.generatemu import *


def main():
    warnings.filterwarnings('ignore')
    
    #====== get the arguments ========# 
    # recording and saving parameters
    record_break  = args.record_break
    record_max = args.record_max
    record_time = lambda t: record_break*t
    nosave = args.nosave
    noprint = args.noprint
    noplot = args.noplot
    trial = args.trial
    direc = args.dir + '/' + args.mu_type

    # algo related parameters
    no_arms = args.no_arms
    no_output = args.no_output
    samp_type = args.samp_type
    reward_type = args.reward_type
    stop_type = args.stop_type
    conf_level = args.conf_level
    precision_level = args.precision_level
    trade_off = args.trade_off
    auto_stop = args.auto_stop
    
    # time variaiton related parameters
    is_timevar = args.is_timevar
    timevar_type = args.timevar_type
    time_decay = args.time_decay
    time_window = args.time_window

    # experiments related parameters
    mu_type = args.mu_type
    mu_list = str2list(args.mu_list, 'float')
    baseline = args.baseline
    mu_time_list = generate_mu_timelist(record_max, no_arms, mu_type, is_timevar, timevar_type, reward_type, mu_list, direc, plot =True)
    baseline_time = lambda t: baseline

    # define the file name as incoporating the  parameter settings
    filename = 'MAB_noarms%d_nooutput%d_%s_%s_%s_trade_off%.2f_conf_level%.2f_precision_level%.2f_break%d_max%d' %\
                (no_arms, no_output, samp_type, reward_type, stop_type, trade_off, conf_level, precision_level, record_break, record_max)  

    if 'Base' in stop_type:   
        filename += '_baseline%4f'%(baseline) 

    if is_timevar:   
        filename += '_timevar%s'%(timevar_type) 
        if timevar_type =='General':
            filename += '_timedecay%.4f'%(time_decay)
        if timevar_type =='Abrupt':
            filename += '_timewindow%.4f'%(time_window)
    filename += '_trial%d'%trial
                
    # Run experiment if the corresponding data doesn't exist yet
    if not os.path.exists("%s/%s.pkl"%(direc, filename)):
        if not is_timevar:
            print('Running trial %d with %s sampling rule and trade off parameter %.2f without time variation...'%(trial, samp_type, trade_off))
        else:
            print('Running trial %d with %s sampling rule and trade off parameter %.2f with %s time variation...'%(trial, samp_type, trade_off, timevar_type))
        
        # initialize the bandit 
        mab_process = runsingle(mu_list = mu_time_list, 
             no_arms = no_arms, no_output = no_output, 
             conf_level = conf_level, trade_off = trade_off, precision_level = precision_level,
             reward_type = reward_type, samp_type = samp_type, stop_type = stop_type, 
             baseline = baseline, 
             record_time = lambda t: record_break*t, record_max = record_max, auto_stop = False, 
             is_timevar = is_timevar, timevar_type = timevar_type, time_decay = time_decay, time_window= time_window,
             verbose = (not noprint))

        # run the bandit
        mab_process.run()
        print('Whether reached stopping: %r' % mab_process.record_stop)
        print('Finished!')
        if not nosave:  
            print('Saving results...')  
            mab_process.save(direc, filename)
        if not noplot:
            if not os.path.exists(direc + '/plots'):
                os.makedirs(direc+'/plots')
            plot_filenames = [file for file in os.listdir(direc+'/plots') if file.startswith(filename)]    
            
            if plot_filenames==[]:
                plot_MAB(mab_process.mabdat, direc, filename)            
    else:
        if not is_timevar:
            print('Trial %d with %s sampling rule and trade off parameter %.2f without time variation are already run, specify another trial number...'%(trial, samp_type, trade_off))
        else:
            print('Trial %d with %s sampling rule and trade off parameter %.2f with %s time variation are already run, specify another trial number...'%(trial, samp_type, trade_off, timevar_type))
        
        if not noplot:
            print('Plotting...')
            if not os.path.exists(direc + '/plots'):
                os.makedirs(direc+'/plots')
            plot_filenames = [file for file in os.listdir(direc+'/plots') if file.startswith(filename)]    
            
            if plot_filenames==[]:
                with open("%s/%s.pkl"%(direc, filename), 'rb') as input:
                    mabdat = dill.load(input)
                plot_MAB(mabdat, direc, filename)            
           
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    '''=========================================
           synthetic data related parameters
    ========================================='''

    # we offer several different type of true mean of reward to be generated automatically, for details of each type, see generatemu.py
    parser.add_argument('--mu-type', type=str, default = "biggap")
    
    # or we can manually specify the true mean of reward for each arm, based on which we simulate the data
    parser.add_argument('--mu-list', type=str, default = "0.1, 0.2, 0.3, 0.4, 0.5") 
    
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
    parser.add_argument('--samp-type', type = str, default = 'TS') # TS, UCB, uniform
    
    # if the base algorithm is not uniform, then we can additionally specify a trade-off parameter (in [1,\infty]) to baseline the extend of exploring sub-topmal arms, the higher it is, the more we will explore sub-optimal arms
    parser.add_argument('--trade-off', type = float, default = 2) 
    
    # what kind of stopping rule, i.e. what kind of testing that you want to reach significance
    parser.add_argument('--stop-type', type = str, default = 'Best') # Best, Best-Base, FWER-Base, FDR-Base

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
    parser.add_argument('--time-decay', type = float, default = 0.001) 

    # if there is time variation, and the type of time variation is '', then we will use change detector based on the testing the difference between moving window, and the window size should be set as approximately sigma^2/(mu_gap)^2, i.e. 2000 for 'biggap' mu_type, 200 for 'biggap' mu_type
    parser.add_argument('--time-window', type = int, default = 200) 
    
    # wheter to stop automatically according to testing criteria
    parser.add_argument('--auto-stop', action = 'store_true') 
    
    '''=========================================
           demonstration related parameters
    ========================================='''
    # over each how many pulls/samples, we record the stauts of bandits
    parser.add_argument('--record-break', type = int, default = 1000)
    
    # the maximum pulls/samples that we run the bandits
    parser.add_argument('--record-max', type = int, default = 10**5)
    
    # how many repitition/trials we run one bandits
    parser.add_argument('--trial', type = int, default = 2)
    
    # whether print out the results or not
    parser.add_argument('--noprint', action='store_true')
    
    # whether plot the results or not
    parser.add_argument('--noplot', action='store_true')
    
    # whether save the results or not
    parser.add_argument('--nosave', action='store_true')
    
    # the directory to save all the plots/data for this experiments
    parser.add_argument('--dir', type = str, default = './simulations')

    args = parser.parse_args()
    logging.info(args)
    main()

