import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate, ceil
from numpy.random import randn, rand, choice
np.set_printoptions(precision = 4)
from scipy.stats import norm, bernoulli, cauchy
# from scipy.stats import bernoulli
#import ipdb
import time

# Import Best arm procedures
import confidence_bounds
import LUCB
import UniformSampling


nounif = 2 # Make sure it's even or it'll break!


def bin_search(a, b, f, MAX, precision = 1e-8):
    # If for max the bound b is already unsatisfied, don't try
    # MAX indicates whether want to find max or min that satisfies f
    if MAX:
        if f(b):
            return 1
        else:
            while b-a > precision:
                pivot = (a+b)/2.
                if f(pivot):
                    a = pivot
                else:
                    b = pivot
    else:
        while b-a > precision:
            pivot = (a+b)/2.
            if f(pivot):
                b = pivot
            else:
                a = pivot
    return pivot

    
'''
Model for one MAB running
'''

class run_single:

    def __init__(self,  no_arms,  m, mu_list, trunctimelist, epsilon = 0,  sigma = 1): 
        # no_arms: number of arms, 
        # epsilon: precision parameter 
        # sigma: if of Gaussian distribution,
        
        self.no_arms = no_arms
        self.m = m 
        self.mu_list = mu_list
        self.reward = 0.
        self.bestidx = argmax(self.mu_list)
        self.trunctimelist = trunctimelist

        self.bestarm = []
        self.rightarm = []
        self.regret = []
        

    def get_results(self, i, alg, trunctime):
        
        # Save best arm
        self.bestarm[i] = alg.mu_hat_decreasing[0]
        self.rightarm[i] = (self.bestarm['index'] == self.bestidx)
        self.regret[i] = alg.reward - self.reward

        print("Bandit at time %d output best arm %d with regret %.2f" % (trunctime, self.rightarm, self.regret))
        
        
    def runmab(self, alpha, epsilon = 0, 
                 bound_type = 'Bernoulli_LIL', samp_name = 'TS', 
                 stop_name = 'Best_k', sigma = 1, 
                 verbose = 0, 
                 control_threshold = float('inf')): # trunctime = max no of pulls
        
        
        trunctime = max(self.trunctimelist)
        self.total_queries = np.zeros(len(self.trunctimelist))
        tt_counter = 0
        
        if verbose:
            print (self.mu_list)

        # --------- Initialize best m arms algorithm to find it---------- #

        alg = MAB(self.m, self.no_arms, delta = alpha, epsilon = epsilon, 
                 bound_type = bound_type, samp_name = samp_name, stop_name = stop_name)
        
        # -----  Run algorithm till stopping rule or truncation time --------#
                   
        while not (alg.should_stop()) | (alg.t > trunctime-1):
            # Get index to pull in this round
            idx = alg.nextpull()
            
            # Pull arm, i.e. get a {Gaussian, Bernoulli} sample
            X = float(rand(len(idx))<self.mu_list[idx]) 
                
            # Record the true best arm
            Xbest = float(rand(len(idx))<self.mu_list[bestidx])
            self.reward += Xbest                        

            # Update arm info (empirical mu, LCB, UCB) for pulled arm
            alg.update(idx, X, 0) 
            
            if (alg.t > trunctimerange[tt_counter]):
                self.get_results(tt_counter, alg, trunctimerange[tt_counter])
                tt_counter = tt_counter + 1
 
        self.get_results(tt_counter, alg, trunctimerange[tt_counter])
