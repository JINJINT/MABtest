
from .uniform_CD import *

class TS_CD(uniform_CD):

    def __init__(self, no_arms: int, no_output: int = 1, 
              conf_level : float =0.05, precision_level: float =0.0, 
              stop_type: str = 'Best', baseline: Callable[[int],float] = None,
              reward_type: str = 'Bernoulli', 
              trade_off: float = 1, prior_para_1: List[float] =None, prior_para_2: List[float] =None,
              time_window: int = 1000, verbose = False) -> None:
        
        # inherent from the uniform class
        super(TS_CD, self).__init__(no_arms, no_output, conf_level, precision_level, stop_type, baseline, reward_type, time_window, verbose)

        # using the uncertainty inflation for trading off optimizaiton and inference
        # the value ranging from 1 to infty, the higher, the bandit algo favors inference more, and incur more regret
        self.trade_off = trade_off 

        # the parameter of the hypothetic distribution of the mean of arms, now we consider the distribution with two parameters, and different arms are independent
        self.prior_para_1: List[float] = prior_para_1 # list of the first parameter of each arm
        self.prior_para_2: List[float] = prior_para_2 # list of the second parameter of each arm


    def __str__(self):
        """ -> str"""
        return self.__class__.__name__ 


    def start(self, restart = False):

        # initiation process that is the same with the uniform bandits
        super(TS_CD, self).start(restart)

        # initiation process that are perticular to the TS bandits
        if not restart:  
            if self.prior_para_1 is None:
                if self.reward_type == 'Bernoulli':
                    self.prior_para_1 = np.repeat(1., self.no_arms)
                if self.reward_type == 'Gaussian':
                    self.prior_para_1 = np.repeat(0., self.no_arms) 
            if self.prior_para_2 is None:
                self.prior_para_2 = np.repeat(1., self.no_arms)    
        else:    
            if self.reward_type == 'Bernoulli':
                self.prior_para_1 = np.repeat(1., self.no_arms)
            if self.reward_type == 'Gaussian':
                self.prior_para_1 = np.repeat(0., self.no_arms)
            self.prior_para_2 = np.repeat(1., self.no_arms) 
     

    def nextpull(self)-> List[int]: 
        '''
        decide which arm(s) to pull next, return a list contains the index of arms
        ''' 

        # Thompson Sampling with the uncertainty inflation (when trade-off ==1, it recovers TS)
        theta = sample_posterior(self.prior_para_1, self.prior_para_2, self.reward_type, self.trade_off)
        argsorted_theta = argsort(theta)[::-1]
        arm_idx = argsorted_theta[:self.no_output] 
        self.next_index = arm_idx      
           
        return list(self.next_index)


    
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
            cb = ConfidenceBound(name = self.bound_type) 
            
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
            
            self.prior_para_1[idx], self.prior_para_2[idx] = update_posterior(self.prior_para_1[idx], self.prior_para_2[idx], rewards[i], self.reward_type)
            
            arm = self.detect_change(arm, reward)
            if arm is None:
                # if there is a change point up to now, restart the whole bandit
                return 
            # if there is no chnage point up to now, continue updating by adding back the corresponding arm with updated info to the sorted dicitonary list                  
            self.mu_hat_decreasing.add(arm)
            self.ucb_decreasing.add(arm)  
            self.lcb_decreasing.add(arm)    

            # update the totoal number of pulls for the bandit    
            self.t += 1
 

 

