
from .uniform import *

class uniform_CD(uniform):

    def __init__(self, no_arms: int, no_output: int = 1, 
              conf_level : float =0.05, precision_level: float =0.0, 
              stop_type: str = 'Best', baseline: Callable[[int],float] = None,
              reward_type: str = 'Bernoulli', 
              time_window: int = 1000, verbose = False) -> None:
        
        # inherent from the uniform class
        super(uniform_CD, self).__init__(no_arms, no_output, conf_level, precision_level, stop_type, baseline, reward_type)

        # the detection window
        self.time_window = time_window
        self.verbose = verbose

    def __str__(self):
        """ -> str"""
        return self.__class__.__name__     


    def start(self, restart = False):
        ''' 
        start/restart (i.e. initiate / re-initiate) the whole bandit

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

        if not restart:
            # if it is not restart, then set the total pulls and accumulated reward to zero, otherwise still keep their value
            self.accreward: float = 0. # the accumulated reward in  total across all the arms
            self.t: int = 0 # number of pulls in total across all the arms  
            self.restart_time = []
        
        # initiate the information for each arm: this process is the same for start or restart
        for i in range(self.no_arms):
            arm = {'index': i, # index of the arm
                   't': 0, # number of pulls of this arm
                   'Sum': 0., # sum of all the reward got so far of this arm
                   'SumSquare':0., # sum of all the reward^2 got so far of this arm
                   'mu_hat': 0., # empirical mean of all the reward got so far of this arm
                   'mu_std':0., # standard derivation of all the reward got so far of this arm
                   'ucb': np.inf, # upper confidence bound of the true mean of this arm using rewards collected so far
                   'lcb': -np.inf, # lower confidence bound of the true mean of this arm using rewards collected so far
                   'mu_front': 0., # empirical mean of all the reward got so far in the 'front' time period (i.e. 0 ~ t-window) of this arm
                   'mu_back': 0., # empirical mean of all the reward got so far in the 'back' time period (i.e. t-window ~ t) of this arm
                   'mu_front_SumSquare':0., # sum of all the reward^2 got so far in the 'front' time period (i.e. 0 ~ t-window) of this arm
                   'mu_back_SumSquare':0., # sum of all the reward^2 got so far in the 'back' time period (i.e. t-window ~ t) of this arm
                   'mu_front_std':0., # standard derivation of all the reward got so far in the 'front' time period (i.e. 0 ~ t-window) of this arm
                   'mu_back_std':0., # standard derivation of all the reward got so far in the 'back' time period (i.e. t-window ~ t) of this arm
                   'break_point':0 # where we split the arrived samples to 'front' and 'back' period
                   }
            self.arms.append(arm)
            self.mu_hat_decreasing.add(arm)
            self.ucb_decreasing.add(arm) 
            self.lcb_decreasing.add(arm)  
        
        # whether the bandit have reach significance to stop, this process is the same for start or restart
        self._should_stop : bool = False 

        # a list of index of arm(s) to pull next
        self.next_index = None



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

            if self.detect_change(arm, reward):
                # if there is a change point up to now, restart the whole bandit
                self.start(restart = True)
                self.restart = True
                self.t += 1
            else:    
                # if there is no chnage point up to now, continue updating by adding back the corresponding arm with updated info to the sorted dicitonary list                  
                self.mu_hat_decreasing.add(arm)
                self.ucb_decreasing.add(arm)  
                self.lcb_decreasing.add(arm)    

            # update the totoal number of pulls for the bandit    
            self.t += 1


    def detect_change(self, arm, reward):
        cb = ConfidenceBound(name = self.reward_type+'LIL')   

        if arm['t']<=self.time_window:
            # if have not accumulate enough samples in total for split into 'front' and 'back' time period, then keeping collecting, and update the statistics related 'front' time period (i.e. 0 ~ t - window )
            arm['mu_front'] = ((arm['t']-1)*arm['mu_front'] + reward) / (arm['t'])
            arm['mu_front_SumSquare'] +=reward**2
            arm['mu_front_std'] = sqrt(abs(arm['mu_front_SumSquare']/arm['t'] - arm['mu_front']**2))
            return False
        else:
            # first update the statistics related to the change point point detection for 'back' time period (i.e t-window ~ t)
            arm['mu_back'] = ((arm['t'] - arm['break_point']-1)*arm['mu_back'] + reward) / (arm['t'] - arm['break_point'])
            arm['mu_back_SumSquare'] +=reward**2
            arm['mu_back_std'] = sqrt(abs(arm['mu_back_SumSquare']/(arm['t'] - arm['break_point']) - arm['mu_back']**2))
            
            # if there is enough samples in both 'front' and 'back' time period, then we do a change detection
            if mod(arm['t'], self.time_window)==0:  
                thred = cb.upper(arm['mu_front'], self.conf_level, arm['break_point'], sigma = arm['mu_front_std']) + cb.upper(arm['mu_back'], self.conf_level, arm['t'] - arm['break_point'], sigma = arm['mu_back_std']) - arm['mu_front'] - arm['mu_back']
                diff = abs(arm['mu_back'] - arm['mu_front'])
                if diff > thred:
                    if self.verbose:
                        print(f'{Fore.RED}%d{Style.RESET_ALL} Restart time step: '%(self.t))
                        #print(f'{Fore.RED}%d, %.5f, %.5f{Style.RESET_ALL}'%(self.t, diff, thred))
                        self.restart_time.append(self.t)
                    return True
                else: 
                    arm['mu_front'] = ((arm['t']-self.time_window)*arm['mu_front'] + self.time_window*arm['mu_back'])/arm['t']
                    arm['mu_front_SumSquare'] = arm['mu_front_SumSquare'] + arm['mu_back_SumSquare']
                    arm['mu_back'] = 0.
                    arm['mu_back_SumSquare'] = 0.
                    arm['break_point'] = arm['t']
                    return False
                  

        
        
