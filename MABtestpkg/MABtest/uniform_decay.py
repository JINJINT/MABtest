
from .uniform import *

class uniform_decay(uniform):

    def __init__(self, no_arms: int, no_output: int = 1, 
              conf_level : float =0.05, precision_level: float =0.0, 
              stop_type: str = 'Best', baseline: Callable[[int],float] = None,
              reward_type: str = 'Bernoulli', 
              time_decay = 0.001) -> None:
        
        # inherent from the uniform class
        super(uniform_decay, self).__init__(no_arms, no_output, conf_level, precision_level, stop_type, baseline, reward_type)

        # the decay rate 
        self.time_decay = time_decay

    def __str__(self):
        """ -> str"""
        return self.__class__.__name__     


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
            
            # decay the memory for existed samples for all the arms, since in this memory decay senario, our memory decays with 'total time' (i.e. the total number of pulls)
            for j in range(self.no_arms):
                eacharm = self.arms[j]
                self.mu_hat_decreasing.remove(eacharm)
                self.ucb_decreasing.remove(eacharm)
                self.lcb_decreasing.remove(eacharm)
                eacharm['Sum'] *= (1-self.time_decay)
                eacharm['t'] *= (1-self.time_decay) 
                if eacharm['t']>0:
                    eacharm['mu_hat'] = eacharm['Sum']/eacharm['t']
                eacharm['lcb'] = cb.lower_timevar(eacharm['mu_hat'], self.conf_level/float(2.*(self.no_arms-self.no_output)), eacharm['t'], self.time_decay, self.t)
                eacharm['ucb'] = cb.upper_timevar(eacharm['mu_hat'], self.conf_level/float(2.*(self.no_output)), eacharm['t'], self.time_decay, self.t)       
                self.mu_hat_decreasing.add(eacharm)
                self.ucb_decreasing.add(eacharm)
                self.lcb_decreasing.add(eacharm)   

            # get the info for correspodning arm
            arm = self.arms[idx]  
            
            # delete the correspodning arm with old info in the sorted dictionary list
            self.mu_hat_decreasing.remove(arm)
            self.ucb_decreasing.remove(arm) 
            self.lcb_decreasing.remove(arm) 

            # update the corresponding arm with the newly arrived sample
            arm['Sum'] += reward
            arm['SumSquare'] += reward**2
            arm['t'] += 1
            arm['mu_hat'] = arm['Sum']/arm['t']
            arm['mu_std'] = sqrt(abs(arm['SumSquare']/arm['t'] - arm['mu_hat']**2))
            arm['lcb'] = cb.lower_timevar(arm['mu_hat'], self.conf_level/float(2.*(self.no_arms-self.no_output)), arm['t'], self.time_decay, self.t)
            arm['ucb'] = cb.upper_timevar(arm['mu_hat'], self.conf_level/float(2.*(self.no_output)), arm['t'], self.time_decay, self.t)

            # add back the corresponding arm with updated info to the sorted dicitonary list                  
            self.mu_hat_decreasing.add(arm)
            self.ucb_decreasing.add(arm)  
            self.lcb_decreasing.add(arm)    

            # update the totoal number of pulls for the bandit    
            self.t += 1



















