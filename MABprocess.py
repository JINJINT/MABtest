import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate, ceil
from numpy.random import randn, rand, choice
from scipy.stats import norm, bernoulli, cauchy
from colorama import Fore, Style
import operator

from MAB import *
from toimport import *
from plotting import *

np.set_printoptions(precision = 10)
   
'''
Model for one MAB running
'''

class MABprocess(object):
    def __init__(self, no_arms, m, mu_list, verbose = False,
                 delta = 0.05, beta = 0.5, epsilon = 0,
                 bound_type = 'Bernoulli_LIL', samp_type= 'TS', stop_type = 'Best_m',
                 record_time = lambda t: 1000*t, record_max = 10**6,
                 auto_stop = False): 

        self.mabdat = MABdat(no_arms, m, mu_list, delta, beta, epsilon,
                 bound_type, samp_type, stop_type, auto_stop)
        
        self.verbose = verbose
        self.record_time = record_time
        self.record_max = record_max 
        self.record_numbers = 1


    def run_MAB(self): 
        
        # --------- Initialize the algorithm ---------- #

        alg = MAB(self.mabdat.no_arms, self.mabdat.m, self.mabdat.delta, self.mabdat.epsilon, self.mabdat.beta, a = None, b = None,
                        bound_type = self.mabdat.bound_type, samp_type = self.mabdat.samp_type, stop_type = self.mabdat.stop_type)
        
        # -----  Run algorithm till stopping rule or truncation time --------#
        self.mabdat.record_time_list = [np.ceil(self.record_time(self.record_numbers))]
               
        while not (self.mabdat.should_stop and self.mabdat.auto_stop) | ((alg.t > self.record_max - 1) and not self.mabdat.auto_stop):
            
            # Get index to pull in this round
            idx = alg.nextpull() 
            # Pull arm, i.e. get a {Gaussian, Bernoulli} sample
            X = float(rand()<self.mabdat.mu_list[idx]) 
            # Update arm info (empirical mu, LCB, UCB) for pulled arm
            alg.update([idx], [X]) 
                
            # Track the optimal solution
            Xbest = float(rand()<self.mabdat.mu_list[self.mabdat.bestidx])
            self.mabdat.accreward_optimal += Xbest 
            
            if (alg.t > self.mabdat.record_time_list[-1]):
                # record the results
                self.record_MAB(alg, self.mabdat.record_time_list[-1]) 
                # update the record time list
                self.record_numbers += 1
                self.mabdat.record_time_list.append(self.record_time(self.record_numbers)) 

            # Record whether the algorihtm should automatically stop
            self.mabdat.should_stop = alg.should_stop()
            
            # Record the info at the stopping time
            if self.mabdat.should_stop:
                self.mabdat.should_stop_time = alg.t  
                self.mabdat.bestarm_stop = alg.mu_hat_decreasing[0]
                self.mabdat.rightarm_stop = (self.mabdat.bestarm_stop['index'] == self.mabdat.bestidx)
                self.mabdat.accreward_stop = alg.accreward
                self.mabdat.allarms_stop = alg.arms
                self.mabdat.accreward_stop = alg.accreward
                self.mabdat.accregret_stop = self.mabdat.accreward_optimal - alg.accreward

        # record the final one
        self.record_MAB(alg, self.mabdat.record_time_list[-1])


    def record_MAB(self, alg, record_time):
        
        # Save best arm
        bestarm = alg.mu_hat_decreasing[0]

        for key in bestarm.keys():
            if key in self.mabdat.bestarm:
                self.mabdat.bestarm[key].append(bestarm[key])
            else:
                self.mabdat.bestarm[key] = [bestarm[key]]   

        accregret = self.mabdat.accreward_optimal - alg.accreward
        self.mabdat.accreward_list.append(alg.accreward)
        self.mabdat.accreward_optimal_list.append(self.mabdat.accreward_optimal)
        
        # save all the arms
        for idx in range(self.mabdat.no_arms):
            arm = alg.arms[idx]
            for key in bestarm.keys():
                if key != 'index':
                    if key in self.mabdat.allarms[idx]:
                        self.mabdat.allarms[idx][key].append(arm[key])
                    else:
                        self.mabdat.allarms[idx][key] = [arm[key]] 


        if self.verbose:
            if self.mabdat.should_stop:
                print(f'{Fore.GREEN}Time: %d,  Estimated Best arm: %d, Averaged regret: %.5f, LCB :%.5f, UCB: %.5f{Style.RESET_ALL}'%\
                 (record_time, bestarm['index'],
                     accregret/record_time, bestarm['lcb'], bestarm['ucb']))

            else:    
                print("Time: %d,  Estimated Best arm: %d, Averaged regret: %.5f, LCB :%.5f, UCB: %.5f"% \
                 (record_time, bestarm['index'], accregret/record_time, bestarm['lcb'], bestarm['ucb']))
    


    def plot_MAB(self, direc, filename, bound = True, arms_idx = None):
        # self.mabdat: a list of objects from class single_run
        # direc: the directory to save the plots
        # arms_idx: which arms to plot

        record_times = self.mabdat.record_time_list

        #=========== plot mean

        if arms_idx is None:
            arms_idx = list(range(self.mabdat.no_arms))

        if len(color_list) < len(arms_idx):
            plot_col = randcolor(len(arms_idx))
        else:
            plot_col = color_list    

        labels = ['arm %d, $\mu$ = %.5f' % (idx, self.mabdat.mu_list[idx]) for idx in arms_idx]   
        bestidx = arms_idx.index(self.mabdat.bestidx)
        labels[bestidx] += ' (best)' 
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
            
        for i, idx in enumerate(arms_idx):
            lcb_list = self.mabdat.allarms[idx]['lcb']
            ucb_list = self.mabdat.allarms[idx]['ucb']
            mu_list = self.mabdat.allarms[idx]['mu_hat']

            if idx == self.mabdat.bestidx: 
                color = 'black'
            else:
                color = plot_col[i % len(plot_col)]   

            ax.errorbar(record_times, mu_list, color = color, linestyle = '-', lw= 1, label=labels[i])
            if bound: 
                ax.errorbar(record_times, lcb_list, color = color, linestyle = '-.', lw= 0.5)
                ax.errorbar(record_times, ucb_list, color = color, linestyle = '-.', lw= 0.5)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                    ncol = 1, prop={'size': 10})
                           
        ax.set_xlabel('Number of pulls', labelpad=7)
        ax.set_ylabel('Estimated mean and confidence bounds', labelpad=7)
        ax.set_xlim((record_times[0], record_times[-1]))
        ax.set_ylim((0, 1))

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))    

        # specially mark the stopping time
        if self.mabdat.auto_stop and self.mabdat.should_stop:
            ax.annotate('Stop', 
                xy=(self.mabdat.should_stop_time, 0), 
                xytext=(self.mabdat.should_stop_time, -0.1), 
                arrowprops = dict(facecolor='black', shrink=0.05))

        filename += '_mean'
        if bound:
            filename += 'withbound'
        saveplot(direc, filename, ax)
        

        # ====== plot reward
    
        fig = plt.figure()
        ax = fig.add_subplot(111)

        reward_list = self.mabdat.accreward_list
      
        ax.errorbar(record_times, reward_list, marker = '.', linestyle = '-.', lw= 2, markersize = 4)

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))                           
        ax.set_xlabel('Number of pulls', labelpad=7)
        ax.set_ylabel('Reward', labelpad=7)
        ax.set_xlim((record_times[0], record_times[-1]))

        # specially mark the stopping time
        if self.mabdat.auto_stop and self.mabdat.should_stop:
            ax.annotate('Stop', 
                xy=(self.mabdat.should_stop_time, -5), 
                xytext=(self.mabdat.should_stop_time, -5.5), 
                arrowprops = dict(facecolor='black', shrink=0.05))

        filename = filename + '_reward'
        saveplot(direc, filename, ax)



        #======== plot regret

        fig = plt.figure()
        ax = fig.add_subplot(111)

        regret_list = list(map(operator.sub, self.mabdat.accreward_optimal_list, self.mabdat.accreward_list))
      
        ax.errorbar(record_times, regret_list, marker = '.', linestyle = '-.', lw= 2, markersize = 4)

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))                           
        ax.set_xlabel('Number of pulls', labelpad=7)
        ax.set_ylabel('Regret', labelpad=7)
        ax.set_xlim((record_times[0], record_times[-1]))

        # specially mark the stopping time
        if self.mabdat.auto_stop and self.mabdat.should_stop:
            ax.annotate('Stop', 
                xy=(self.mabdat.should_stop_time, -5), 
                xytext=(self.mabdat.should_stop_time, -5.5), 
                arrowprops = dict(facecolor='black', shrink=0.05))

        filename = filename + '_regret'
        saveplot(direc, filename, ax)







    def save_MAB(self, direc, filename):
        saveobject(self.mabdat, direc, filename)



    
#====== object defined to save the mab data

class MABdat(object): 
    def __init__(self, no_arms, m, mu_list, 
                 delta = 0.05, beta = 0.5, epsilon = 0,
                 bound_type = 'Bernoulli_LIL', samp_type= 'TS', stop_type = 'Best_m',
                 auto_stop = False): 

        #====== parameters related to input data      

        self.no_arms = no_arms # number of arms 
        self.m = m # the number of best m arms 
        self.mu_list = mu_list # the true mean of the arms
        self.bestidx = argmax(self.mu_list) # index of the true best arm

        #======= parameters related to the algorithm configuration

        self.beta = beta # the trade off parameter in TTTS sampling
        self.delta = delta # the confidence level
        self.epsilon = epsilon # the epsilon parameter 
        self.bound_type = bound_type
        self.samp_type = samp_type
        self.stop_type = stop_type 

        #======= parameters related to the recorded time points 

        self.record_time_list = [] # the record time points     

        #======= parameters related to recorded information
        
        self.bestarm = {} # record the best arm info choosed by the algo at each step
        self.allarms = [{} for i in range(self.no_arms)] # record all the arm info 
        
        self.accreward_list = [] # accumulated reward at each step        
        self.accreward_optimal = 0. # accumulated reward
        self.accreward_optimal_list = [] # accumulated optimal reward at each step
        
        #======= parameters related to the stopping time

        self.auto_stop = auto_stop # whether the alg is allowed to stop automatically
        self.should_stop = False # whether the alg should stopped according to its stop rule
        self.should_stop_time = -1 # record the alg automated stopping time, None represent the alg stopped by forced tructime
        
        #======= record the info at stopping time

        self.bestarm_stop = None 
        self.rightarm_stop = None
        self.accreward_stop = None
        self.allarms_stop = None
        self.accreward_stop = None
        self.accregret_stop = None          



















