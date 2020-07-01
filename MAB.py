import math
from numpy import sqrt, log, exp, mean, cumsum, zeros, argsort, argmin, argmax, array, random
import numpy as np
from sortedcontainers import SortedListWithKey
from confidence import ConfidenceBound
from numpy.random import randn, rand, choice, randint, beta, binomial

from estimation import *

np.set_printoptions(precision=10)

class MAB(object):

    def __init__(self, K, m = 1, delta=0.05, epsilon=0.0, beta = 0.5, a = None, b = None,
             bound_type='Bernoulli_LIL', samp_type="TS", stop_type = "Best_m"):
        
        self.samp_type = samp_type # TTTS; inflated-UCB; 
        self.stop_type = stop_type # Best arm; GN; FDR 
        self.bound_type = bound_type # sub-G LIL, mPRST

        self.beta = beta # trade off parameter in TTTS
        
        self.a = a # priors for the mean
        self.b = b # priors for the mean 

        self.K = K # number of arms
        self.m = m # top m
    
        self.delta = delta # confidence level
        self.epsilon = epsilon # precision

        self.start()

    def start(self):
        self._should_stop = False
        self.next_index = None

        # arms list contains the original indeces (of the mu_list)
        self.arms = []
        self.mu_hat_decreasing = SortedListWithKey(key = lambda x: -x['mu_hat'])
        self.ucb_decreasing = SortedListWithKey(key = lambda x: -x['ucb'])
        self.accreward = 0.
        self.t = 0 # time

        if self.a is None:
            self.a = np.repeat(1, self.K)
        if self.b is None:
            self.b = np.repeat(1, self.K)
    
        for i in range(self.K):
        # 'T' number of times the arm was pulled
            arm = {'index': i, 'Sum': 0., 't': 0, 'mu_hat': 0., 'ucb': 1., 'lcb': 0., 'a': self.a[i], 'b': self.b[i]}
            self.arms.append(arm)
            self.mu_hat_decreasing.add(arm)
            self.ucb_decreasing.add(arm)


    def should_stop(self): # stopping rule
        # Boolean indicating whether sampling should stop or not.   

        if self.stop_type == "Best_m":
        # stop when find the best m with confidence'''

            # find the best m arm with the highest mean
            top_m = self.mu_hat_decreasing[0:self.m]    
            # sort them by lcb
            top_m = sorted(top_m, key = lambda y: y['lcb'])    

            # within the rest k-m arms find the one with the highest UCB
            maxrest_idx = 0
            while self.ucb_decreasing[maxrest_idx] in top_m:
                maxrest_idx += 1
            maxrest = self.ucb_decreasing[maxrest_idx]

            # stop if the lowerst lcb of the best m arms is higher than the highest ucb of the rest m-k best arm
            if top_m[0]['lcb'] > maxrest['ucb'] - self.epsilon:
                self._should_stop = True

        return self._should_stop    


    def nextpull(self): 
        # decide which arm to pull next
        # return a vector contains the id of arms

        if self.samp_type == "TS":
            theta = [beta(self.a[i], self.b[i]) for i in range(self.K)]
            arm_idx = argmax(theta)
            if self.beta > 0:
                if binomial(1, 1-self.beta) ==1 :
                    newarm_idx = arm_idx
                    while newarm_idx == arm_idx:
                        thetanew = [beta(self.a[i], self.b[i]) for i in range(self.K)]
                        thetanew[arm_idx] = 0.
                        newarm_idx = argmax(thetanew)
                    arm_idx = newarm_idx
            
            self.next_index  = arm_idx

        if self.samp_type == "uniform":
            self.next_index = randint(0, self.K)

        return self.next_index 


    def update(self, arms, rewards):
        
        # update the arm metric estimate
        for i, idx in enumerate(arms):
            arm = self.arms[idx]
            reward = rewards[i]
            self.accreward += reward 

            self.mu_hat_decreasing.remove(arm)
            self.ucb_decreasing.remove(arm)

            arm['Sum'] += reward
            arm['t'] += 1.
            arm['mu_hat'] = arm['Sum']/arm['t']

            cb = ConfidenceBound(name = self.bound_type)     
            arm['lcb'] = cb.lower(arm['mu_hat'], self.delta/float(2.*(self.K-self.m)), arm['t'])
            arm['ucb'] = cb.upper(arm['mu_hat'], self.delta/float(2.*(self.m)), arm['t'])

            if self.samp_type == "TS":
                arm['a'], arm['b'] = posterior(arm, reward)
                self.a[idx] = arm['a']
                self.b[idx] = arm['b']

            self.t += 1
           
            self.mu_hat_decreasing.add(arm)
            self.ucb_decreasing.add(arm)


