# -*- coding: utf-8 -*-
""" The AdBandits bandit algorithm, mixing Thompson Sampling and BayesUCB.
- Reference: [AdBandit: A New Algorithm For Multi-Armed Bandits, F.S.Truzzi, V.F.da Silva, A.H.R.Costa, F.G.Cozman](http://sites.poli.usp.br/p/fabio.cozman/Publications/Article/truzzi-silva-costa-cozman-eniac2013.pdf)
- Code inspired from: https://github.com/flaviotruzzi/AdBandits/
.. warning:: This policy is very not famous, but for stochastic bandits it works usually VERY WELL! It is not anytime thought.
"""

import numpy as np
import math
from numpy import sqrt, log, exp, mean, cumsum, zeros, argsort, argmin, argmax, array, random, argwhere, mod
from numpy.random import randn, rand, choice, randint, beta, binomial, multinomial
from sortedcontainers import SortedListWithKey
from colorama import Fore, Style
import itertools
from typing import Any, Callable, List

from .confidence import *
from .estimation import *
from .toimport import *

class uniform():

    def __init__(self, no_arms: int, no_output: int = 1, 
              conf_level : float =0.05, precision_level: float =0.0, 
              stop_type: str = 'Best', baseline: Callable[[int],float] = None,
              reward_type: str = 'Bernoulli') -> None:
        
        # number of arms in total
        self.no_arms: int = no_arms

        # how many arms are interested as final output 
        self.no_output: int = no_output
        
        # the type of auto stopping rule: 
        #    1. 'Best': find the best m arms with confidence; 
        #    2. 'Best-Base': find the best m arms over baseline with confidence; if not exsited, return baseline
        #    3. 'FWER-Base': find m positive arm over baseline with FWER control; if not exsited, return baseline
        #    4. 'FDR-Base': find m positive arms over baseline with FDR control; if not exsited, return baseline
        self.stop_type: str = stop_type 
        
        # what distribution of the reward for each arm follows 
        # (for now we assume all the arms have the same reward type, though this can be modified to different typefor different arm)
        self.reward_type: str = reward_type         
    
        # confidence level, usually set at 0.05
        self.conf_level: float = conf_level 

        # precision level, usually set at 0, it requires more rigid (when >0) or more loose (when <0) of the stopping rule, that is:
        #    1. 'Best': (mean of each best k arms) - (mean of rest arms) > precision 
        #    2. 'Best-Base': (mean of each best k arms) - (mean of rest arms) > precision  and   (mean of best k arms) -baseline > precision 
        #    3. 'FWER-Base': (mean of each choosen k arms) - baseline > precision, with FWER control
        #    4. 'FDR-Base': (mean of each choosen k arms) - baseline > precision, with FDR control
        self.precision_level: float = precision_level 
        
        # the baseline: a function of time, represent the true mean of the baseline arm over time
        # (for now we take the truth of baseline as known, and do not sample for it during the bandit algo; 
        #  though this can be modified to treat it as unkown, and sample for it during the bandit algo.)
        self.baseline: Callable[[int],float] = baseline
        
        # the index of arms that are outputed in the end
        self.output = None 

    
    def __str__(self):
        """ -> str"""
        return self.__class__.__name__    


    def start(self):
        ''' 
        start (i.e. initiate) the whole bandit

        '''
        
        # a list of dictionary, to contain dynamic information of all arms
        self.arms = [] 
        
        # a list of sorted dictionary, where each dictionary will represent one arm, and the ordering corresponds to decreasing mu_hat (i.e. the empirical mean of the reward) 
        # this hashtable structure allow O(1) time to insert/delete elements while keeping the ordering according to their mu_hat 
        self.mu_hat_decreasing = SortedListWithKey(key = lambda x: -x['mu_hat']) 
        
        # a list of sorted dictionary, where each dictionary will represent one arm, and the ordering corresponds to decreasing ucb (i.e. the upper confidence bounds of the mean of reward) 
        self.ucb_decreasing = SortedListWithKey(key = lambda x: -x['ucb']) 

        # a list of sorted dictionary, where each dictionary will represent one arm, and the ordering corresponds to decreasing lcb (i.e. the lower confidence bounds of the mean of reward) 
        self.lcb_decreasing = SortedListWithKey(key = lambda x: -x['lcb']) 

        self.accreward: float = 0. # the accumulated reward in  total across all the arms
        self.t: int = 0 # number of pulls in total across all the arms  
        
        # initiate the information for each arm: this process is the same for start or restart
        for i in range(self.no_arms):
            arm = {'index': i, # index of the arm
                   't': 0, # number of pulls of this arm
                   'Sum': 0., # sum of all the reward got so far of this arm
                   'SumSquare':0., # sum of all the reward^2 got so far of this arm
                   'mu_hat': 0., # mean of all the reward got so far of this arm
                   'mu_std':0., # standard derivation of all the reward got so far of this arm
                   'ucb': np.inf, # upper confidence bound of the true mean of this arm using rewards collected so far
                   'lcb': -np.inf # lower confidence bound of the true mean of this arm using rewards collected so far
                   }
            self.arms.append(arm)
            self.mu_hat_decreasing.add(arm)
            self.ucb_decreasing.add(arm) 
            self.lcb_decreasing.add(arm)  
        
        # whether the bandit have reach significance to stop, this process is the same for start or restart
        self._should_stop : bool = False 

        # a list of index of arm(s) to pull next
        self.next_index = None



    def should_stop(self)-> bool: 
        # return a boolean indicating whether the bandit should stop or not, according to whether have reached significance for the speficific testing problem

        if 'Best' in self.stop_type:  

            # find the best m arm with the highest mean
            topm = self.mu_hat_decreasing[0:self.no_output]    
            # sort them by lcb
            topm = sorted(topm, key = lambda y: y['lcb'])
            # find the one with the lowest lcb within the top m
            minlcb_topm = topm[0]  

            # find the best one among all the arms
            maxucb_arm = self.ucb_decreasing[0]  

            # within the rest k-m arms find the one with the highest UCB
            maxucb_rest_idx = 0
            while self.ucb_decreasing[maxucb_rest_idx] in topm:
                maxucb_rest_idx += 1
            maxucb_rest = self.ucb_decreasing[maxucb_rest_idx]  

            if self.stop_type == "Best":
                ''' 
                stop when find the best m with confidence
                '''
                if minlcb_topm['lcb'] > maxucb_rest['ucb'] - self.precision_level:
                    self._should_stop = True
                    self.output = [top['index'] for top in topm] 

            if self.stop_type == "Best-Base":
                ''' 
                stop when find no one is better than the baseline by precision_level, or when find there is at least m that is better than control and return the best m
                '''
                
                # Stop if no arm is precision_level better than the baseline, and output empty set
                if self.baseline(self.t) > maxucb_arm['ucb'] - self.precision_level:
                    self._should_stop = True
                    self.output = [] 
                
                # Stop if there are at least m arms is better than the baseline by precision_level, and outout the top m among them
                elif (minlcb_topm['lcb'] > maxucb_rest['ucb']- self.precision_level) and (minlcb_topm['lcb'] > self.baseline(self.t) + self.precision_level):
                    self._should_stop = True
                    self.output = [top['index'] for top in topm] 

        
        if self.stop_type == 'FWER-Base':
            ''' 
            stop when find no one is better than the baseline by precision_level, or when find there is at least m that is better than control and return m of them
            '''
            
            # find the one with highest upper confidence bound among all the arms
            maxucb_arm = self.ucb_decreasing[0] 
            
            # find the top m ones with highest lower confidence bound among all the arms
            minlcb_topm = self.lcb_decreasing[self.no_output-1]  

            # Stop if no arm is better than the baseline by precision level, and output empty set
            if self.baseline(self.t) > maxucb_arm['ucb'] - self.precision_level:
                self._should_stop = True
                self.output = [] 
            
            # Stop if there are at least m arms is better than the baseline by precision level, and outout the first m among them
            elif minlcb_topm['lcb'] > self.baseline(self.t) + self.precision_level:
                self._should_stop = True
                self.output = [positive['index'] for positive in self.lcb_decreasing[:self.no_output]]             


        if self.stop_type == "FDR-Base":
            ''' 
            stop when find there are at least m that is better than control by precision level and output all of them
            '''
            
            # get the corrected confidence level under dependency (due to adaptive sampling)
            conf_level_prime = float(self.conf_level/(6.4*log(36/self.conf_level)))
            
            # get the upper confidence bound for FDR control
            cb = ConfidenceBound(name = self.reward_type+'LIL')  
            
            # using BH method (https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini–Hochberg_procedure) to select positive arms over baseline by precision level with FDR control
            Selected_BH = np.array([])
            k = self.no_arms
            while np.sum(Selected_BH > self.baseline(self.t)) <= k and k>0:
                Selected_BH = np.array([cb.lower(arm['mu_hat'], float(conf_level_prime*k/(2.*(self.no_arms))), arm['t']) for i, arm in enumerate(self.arms)])
                Selected_BH[Selected_BH == np.nan] = -np.inf
                k = k-1   
            
            if np.sum(Selected_BH > self.baseline(self.t))>0:
                self.output = np.concatenate(argwhere(Selected_BH > self.baseline(self.t)))   
            
            # Stop when there are more than m arms selected
            if self.output is not None:
                if len(self.output) >= self.no_output:
                    self._should_stop = True

        return self._should_stop  
 

    def nextpull(self)-> List[int]: 
        '''
        decide which arm(s) to pull next, return a list contains the index of arms
        '''

        # random/uniform sampling
        self.next_index = randint(0, self.no_arms, self.no_output) 
           
        return self.next_index 


    def update(self, arms: list, rewards: list)-> None:
        '''
        Update the bandit given the observed rewards for arms that are just pulled
        
        Arguments:
            arms : contains the index of pulled arms
            rewards : the observed reward for each arm in arms
        '''
        if type(arms) is not list:
            arms = [arms]
        if type(rewards) is not list:    
            rewards = [rewards]
        if len(arms)!=len(rewards):
            raise ValueError("The length of pulled arms and rewards should be equal!") 
        
        for i, idx in enumerate(arms):
            
            # get the reward for correspodning arm
            reward = rewards[i] 
            # update the accumulated reward in total for the bandit
            self.accreward += reward 
            
            # initiate the object for caculating confidence bounds
            cb = ConfidenceBound(name = self.reward_type + 'LIL') 
            
            # get the info dictionary for correspodning arm
            arm = self.arms[idx]  
            # delete the correspodning arm with old info in the sorted dictionary list
            self.mu_hat_decreasing.remove(arm)
            self.ucb_decreasing.remove(arm) 
            self.lcb_decreasing.remove(arm) 
            
            # calculate the updated info for the corresponding arm 
            arm['Sum'] += reward
            arm['SumSquare'] += reward**2
            arm['t'] += 1
            arm['mu_hat'] = arm['Sum']/arm['t']
            arm['mu_std'] = sqrt(abs(arm['SumSquare']/arm['t'] - arm['mu_hat']**2))
            arm['lcb'] = cb.lower(arm['mu_hat'], self.conf_level/float(2.*(self.no_arms-self.no_output)), arm['t'], sigma =arm['mu_std'])
            arm['ucb'] = cb.upper(arm['mu_hat'], self.conf_level/float(2.*(self.no_output)), arm['t'], sigma =arm['mu_std'])  
            
            # add back the corresponding arm with updated info to the sorted dicitonary list                  
            self.mu_hat_decreasing.add(arm)
            self.ucb_decreasing.add(arm)  
            self.lcb_decreasing.add(arm)    

            # update the totoal number of pulls for the bandit    
            self.t += 1
        

        

                        




            

















