import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate, ceil
from numpy.random import randn, rand, choice, normal
from scipy.stats import norm, bernoulli, cauchy
from colorama import Fore, Style
import operator

#==== algorithms in stationary environments =======#
# naive random/uniform sampling
from .uniform import uniform
# adaptive sampling with trade-off parameter
from .UCB import TS
from .TS import TS

#==== algorithms trying to adapt to dynamically changing environments
# --- unkown/general changes

# naive random/uniform sampling
from .uniform_decay import uniform_decay
# adaptive sampling with trade-off parameter
from .UCB_decay import UCB_decay
from .TS_decay import TS_decay

# --- abrupt changes

# naive random/uniform sampling
from .uniform_CD import uniform_CD
# adaptive sampling with trade-off parameter
from .UCB_CD import UCB_CD
from .TS_CD import TS_CD

#==== helper functions =======#
from .confidence import *
from .estimation import *
from .toimport import *

#==== experiments function ====#
from .generatemu import *
from .plotting import *

   
'''
Run a single MAB once
'''

class runsingle():
    def __init__(self, mu_list: Callable[[int, int],float], 
                 no_arms: int, no_output: int, 
                 conf_level: float = 0.05, trade_off: float = 1, precision_level: float = 0.,
                 reward_type: str = 'Bernoulli', samp_type: str = 'TS', stop_type: str = 'Best', 
                 baseline: Callable[[int], float] = None, 
                 record_time : Callable[[int],float] = lambda t: 1000*t, record_max: int = 10**6, auto_stop : bool = False, 
                 is_timevar: bool = False, timevar_type: str= 'General', time_decay : float = 0.001, time_window: int = 1000,
                 verbose = False): 

        self.mabdat: MABdat = MABdat(mu_list, no_arms, no_output, conf_level, trade_off, precision_level,
                 reward_type, samp_type, stop_type, auto_stop, baseline, is_timevar, timevar_type, time_decay, time_window)
         
        self.verbose: bool = verbose
        self.record_time: Callable[[int],float] = record_time
        self.record_max: int = record_max 
        self.record_numbers: int = 1
        self.record_stop: bool = False
        
        # --------- Initialize the algorithm ---------- #
        if is_timevar: # there is no time variation
        
            if samp_type == 'TS':
                alg = TS(no_arms)
            
            if samp_type =='UCB':
                alg = UCB(no_arms)

            if samp_type =='uniform':
                alg = uniform(no_arms)

        else: # there is time variation

            # general unknown time variation
            if timevar_type == 'General': 

                if samp_type == 'TS':
                    alg = TS_decay(no_arms)
                
                if samp_type =='UCB':
                    alg = UCB_decay(no_arms)

                if samp_type =='uniform':
                    alg = uniform_decay(no_arms)
            
            # abrupt time variation
            if timevar_type == 'Abrupt':

                if samp_type == 'TS':
                    alg = TS_CD(no_arms)
                
                if samp_type =='UCB':
                    alg = UCB_CD(no_arms)

                if samp_type =='uniform':
                    alg = uniform_CD(no_arms)

        alg.no_output = no_output
        alg.conf_level = conf_level
        alg.precision_level = precision_level
        alg.reward_type = reward_type
        alg.stop_type = stop_type
        
        if 'Base' in alg.stop_type:
            alg.baseline = baseline
        
        if samp_type != 'uniform':
            alg.trade_off = trade_off
            if 'TS' in samp_type:
                alg.prior_para_1 = prior_para_1
                alg.prior_para_2 = prior_para_2
            
        if is_timevar:
            if timevar_type =='General' :
                alg.time_decay = time_decay
            if timevar_type =='Abrupt':
                alg.time_window = time_window 

        alg.start()          

        self.alg = alg            


    def run(self): 
        
        # -----  Run algorithm till stopping rule or truncation time --------#
        self.alg.record_time_list = [np.ceil(self.record_time(self.record_numbers))]
        
        # start by pull each arm once
        idx = list(range(self.mabdat.no_arms))
        if self.mabdat.reward_type == 'Bernoulli':
            X = [float(rand()<self.mabdat.mu_list(j+1, i)) for j, i in enumerate(idx)]
        if self.mabdat.reward_type == 'Gaussian':
            X = [float(normal(loc = self.mabdat.mu_list(j+1, i))) for j, i in enumerate(idx)] 
        
        Xbest = []
        for j in range(len(idx)):
            mu_list_now = [self.mabdat.mu_list(alg.t+j+1, i) for i in range(self.mabdat.no_arms)] 
            bestidx_now = argmax(mu_list_now)
            self.mabdat.bestidx.append(bestidx_now)
            if self.mabdat.reward_type == 'Bernoulli':
                Xbest.append(mu_list_now[bestidx_now])
            if self.mabdat.reward_type == 'Gaussian':
                Xbest.append(mu_list_now[bestidx_now])

        self.mabdat.accreward_optimal += np.sum(np.array(Xbest))

        # Update arm info (empirical mu, LCB, UCB) for pulled arm
        self.alg.update(idx, X)
               
        while not (self.mabdat.should_stop and self.mabdat.auto_stop) | ((self.alg.t > self.record_max - 1) and not self.mabdat.auto_stop):
            # Get index to pull in this round
            idx = self.alg.nextpull() 
            # Pull arms, i.e. get {Gaussian, Bernoulli} samples
            if self.mabdat.reward_type == 'Bernoulli':
                X = [float(rand()<self.mabdat.mu_list(self.alg.t+j+1, i)) for j, i in enumerate(idx)]
            if self.mabdat.reward_type == 'Gaussian':
                X = [float(normal(loc = self.mabdat.mu_list(self.alg.t+j+1, i))) for j, i in enumerate(idx)]
                
            # Track the optimal solution
            Xbest = []
            for j in range(len(idx)):
                mu_list_now = [self.mabdat.mu_list(self.alg.t+j+1, i) for i in range(self.mabdat.no_arms)] 
                bestidx_now = argmax(mu_list_now)
                self.mabdat.bestidx.append(bestidx_now)
                if self.mabdat.reward_type == 'Bernoulli':
                    Xbest.append(mu_list_now[bestidx_now])
                if self.mabdat.reward_type == 'Gaussian':
                    Xbest.append(mu_list_now[bestidx_now])               
            self.mabdat.accreward_optimal += np.sum(np.array(Xbest)) 

            # Update arm info (empirical mu, LCB, UCB) for pulled arm
            self.alg.update(idx, X)

            # Save the pulled arm
            self.mabdat.pullarm.append(idx)
            
            if (self.alg.t > self.mabdat.record_time_list[-1]):
                # record the results
                self.record_MAB(self.alg, self.mabdat.record_time_list[-1]) 
                # update the record time list
                self.record_numbers += 1
                self.mabdat.record_time_list.append(self.record_time(self.record_numbers)) 

            # Record whether the algorihtm should automatically stop
            self.mabdat.should_stop = self.alg.should_stop()
            # Save the alg output
            self.mabdat.output.append(self.alg.output)
            
            # Record the info at the stopping time
            if self.mabdat.should_stop and not self.record_stop:
                self.mabdat.should_stop_time = self.alg.t  
                self.mabdat.outputarm_stop = [self.alg.arms[idx] for idx in self.alg.output]
                self.mabdat.accreward_stop = self.alg.accreward
                self.mabdat.allarms_stop = self.alg.arms
                self.mabdat.accregret_stop = self.mabdat.accreward_optimal - self.alg.accreward
                self.record_stop = True

        # record the final one
        self.record_MAB(self.mabdat.record_time_list[-1])


    def record(self, record_time: int):
        
        # Save best arm
        bestarm = self.alg.mu_hat_decreasing[0]

        for key in bestarm.keys():
            if key in self.mabdat.bestarm:
                self.mabdat.bestarm[key].append(bestarm[key])
            else:
                self.mabdat.bestarm[key] = [bestarm[key]]   

        accregret = self.mabdat.accreward_optimal - self.alg.accreward
        self.mabdat.accreward_list.append(self.alg.accreward)
        self.mabdat.accreward_optimal_list.append(self.mabdat.accreward_optimal)
        
        # save all the arms
        for idx in range(self.mabdat.no_arms):
            arm = self.alg.arms[idx]
            for key in bestarm.keys():
                if key != 'index':
                    if key in self.mabdat.allarms[idx]:
                        self.mabdat.allarms[idx][key].append(arm[key])
                    else:
                        self.mabdat.allarms[idx][key] = [arm[key]] 


        if self.verbose:
            if self.mabdat.should_stop:
                print(f'{Fore.GREEN}Trade_off_para: %.2f, Time: %d,  Estimated Best arm: %d, Averaged regret: %.5f, LCB :%.5f, UCB: %.5f, restart: %r {Style.RESET_ALL}'%\
                 (self.mabdat.trade_off, record_time, bestarm['index'],
                     accregret/record_time, bestarm['lcb'], bestarm['ucb'], self.alg.restart))

            else:    
                print("Trade_off_para: %.2f, Time: %d,  Estimated Best arm: %d, Averaged regret: %.5f, LCB :%.5f, UCB: %.5f, restart: %r"% \
                 (self.mabdat.trade_off, record_time, bestarm['index'], accregret/record_time, bestarm['lcb'], bestarm['ucb'], self.alg.restart))
    

    def save(self, direc: str, filename: str):
        savelambda(self.mabdat, direc, filename)


    
#====== object defined to save the mab data

class MABdat(object): 
    def __init__(self, no_arms:int, no_output:int, mu_list: Callable[[int, int],float], 
                 conf_level: float = 0.05, trade_off: float = 1, precision_level: float = 0,
                 reward_type : str = 'Bernoulli', samp_type: str='TS', stop_type: str = 'Best',
                 auto_stop = False, baseline: Callable[[int],float] = None, 
                 is_timevar: bool = False, timevar_type: str='General', time_decay: float = 0.1, time_window: int = 1000): 

        #====== parameters related to input data      

        self.no_arms: int = no_arms # number of arms 
        self.no_output: int = no_output # the number of best m arms 
        self.mu_list: Callable[[int, int],float] = mu_list # the true mean of the arms      
        self.baseline: Callable[[int], float] = baseline
        self.bestidx = []

        #======= parameters related to the algorithm configuration

        self.trade_off: float = trade_off # the trade off parameter in TTTS sampling
        self.conf_level: float = conf_level # the confidence level
        self.precision_level: float = precision_level # the precision_level parameter 
        self.reward_type: str = reward_type
        self.samp_type: str = samp_type
        self.stop_type: str = stop_type 

        #======= parameters related to the recorded time points 

        self.record_time_list: List[int] = [] # the record time points     

        #======= parameters related to recorded information
        
        self.bestarm = {} # record the best arm info choosed by the algo at each step
        self.allarms = [{} for i in range(self.no_arms)] # record all the arm info 
        self.pullarm = [] # record the pulled arms at each time
        self.output = [] # record the output of the hypothesis testing at each time
        
        self.accreward_list = [] # accumulated reward at each step        
        self.accreward_optimal: float = 0. # accumulated reward
        self.accreward_optimal_list = [] # accumulated optimal reward at each step
        
        #======= parameters related to the stopping time

        self.auto_stop: bool = auto_stop # whether the alg is allowed to stop automatically
        self.should_stop: bool = False # whether the alg should stopped according to its stop rule
        self.should_stop_time: int = np.inf # record the alg automated stopping time, None represent the alg stopped by forced tructime
        
        #======= record the info at stopping time

        self.bestarm_stop = None 
        self.rightarm_stop = None
        self.accreward_stop: float = None
        self.allarms_stop = None
        self.accreward_stop: float = None
        self.accregret_stop: float = None      


        #======= info about the time decay
        self.is_timevar = is_timevar
        self.timevar_type = timevar_type
        self.time_decay = time_decay
        self.time_window = time_window


