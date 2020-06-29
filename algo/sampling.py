"""The LUCB algorithm originally proposed in

Kalyanakrishnan, S., Tewari, A., Auer, P., & Stone, P. "PAC subset 
selection in stochastic multi-armed bandits." ICML 2012.

with "improved" LUCB++ version that results in better performance by

Simchowitz, M., Jamieson, K., Recht, B. "Towards a Richer Understanding of 
Adaptive Sampling in the Moderate-Confidence Regime." Preprint 2016.
"""
import time
import math
from numpy import sqrt, log, exp, mean, cumsum, zeros, argsort, argmin, argmax, array, random
import numpy
from sortedcontainers import SortedListWithKey
import ipdb
from confidence_bounds import ConfidenceBound

numpy.set_printoptions(precision=4)
numpy.set_printoptions(linewidth=200)

class MAB(object):
    def __init__(self, m = 1, K, delta=0.05, epsilon=0.0, 
             bound_type='Bernoulli_LIL', samp_name="TS", stop_name = "Best_m"):
        self.sampname = samp_name # TTTS; inflated-UCB; 
        self.stopname = stop_name # Best arm; GN; FDR 
        self.boundtype = bound_type # sub-G LIL, mPRST


        self.beta = beta # trade off parameter in TTTS
        self.alpha = alpha # trade off parameter in inflated UCB

        self.K = K # number of arms
        self.m = m # top m

        self.t = 0 # time
    
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
        self.reward = 0.
    
        # Draw all of them uniformly for 10
    
        for i in range(self.K):
        # 'T' number of times the arm was pulled
            arm = {'index': i, 'Sum': 0., 't': 0, 'mu_hat': 0., 'ucb': float('inf'), 'lcb': -float('inf'), 'a': 1., 'b':1.}
            self.arms.append(arm)
            self.mu_hat_decreasing.add(arm)
            self.ucb_decreasing.add(arm)


    def should_stop(self, stop_threshold=float('inf')): # stopping rule
    """Boolean indicating whether sampling should stop or not.

    Args:
    stop_threshold: if there exist k arms which can confidently be said 
    to have means greater than stop_threshold, method returns True
    """    

        if self.stopname == "Best_m":
        '''stop when find the best m with confidence'''

           # find the best m arm with the highest mean
            top_m = self.mu_hat_decreasing[0:self.m]    
            # sort them by lcb
            top_m = sorted(top_m, key = lambda y: y['lcb'])    

            # within the rest k-m arms find the one with the highest UCB
            maxrest_idx = 0
            while self.ucb_decreasing[maxrest_idx] == top_m:
                maxrest_idx += 1
            maxrest = self.ucb_decreasing[maxrest_idx]

            # stop if the lowerst lcb of the best m arms is higher than the highest ucb of the rest m-k best arm
            if top_m[0]['lcb'] > maxrest[0]['ucb'] - self.epsilon:
                self._should_stop = True

        #if self.stopname == "Best_Control":
        ''' stop when there is no arms that is better than control 
              or 
            find the best one with confidence when there are arms that is better than control'''
    
        #if self.stopname == "FDR":

    return self._should_stop    


    def nextpull(self): 
        # decide which arm(s) to pull next
        # return a vector contains the id of arms
        if self.sampname == "TS":
            theta = [random.beta(a[i], b[i]) for i in range(K)]
            arm_idx = which.max(theta)
            if self.beta>0:
            if random.binomial(self.beta) ==1:
            newarm = arm_idx
            while newarm == arm:
            thetanew = [random.beta(a[i], b[i]) for i in range(K)]
            armnew_idx = which.max(thetanew)
            arm_idx = armnew_idx
            self.next_index  = armnew_idx

        if self.sampname == "uniform":
            self.next_index = random.randint(0, self.K)

    return self.next_index 


    def update(self, arms, rewards, verbose = 0):
        
        # update the arm metric estimate
        for idx in arms:
            arm = self.arms[idx]
            reward = rewards[idx]
            self.reward + = reward 

            self.mu_hat_decreasing.remove(arm)
            self.ucb_decreasing.remove(arm)

            arm['Sum'] += reward
            arm['t'] += 1.
            arm['mu_hat'] = arm['Sum']/arm['t']

            cb = ConfidenceBound(self.bound_type).        
            arm['lcb'] = cb.lower(arm['mu_hat'], self.delta/float(2.*(self.K-self.m)), arm['t'])
            arm['ucb'] = cb.upper(arm['mu_hat'], self.delta/float(2.*(self.m)), arm['t'])

            if self.sampname = "TS":
                arm['a'], arm['b'] = posterior(arm, reward)

            self.arms.append(arm)
            self.t += 1
           
            self.mu_hat_decreasing.add(arm)
            self.ucb_decreasing.add(arm)

