
from .uniform_CD import *


class UCB_CD(uniform_CD):

    def __init__(self, no_arms: int, no_output: int = 1, 
              conf_level : float =0.05, precision_level: float =0.0, 
              stop_type: str = 'Best', baseline: Callable[[int],float] = None,
              reward_type: str = 'Bernoulli', trade_off: float = 1,
              time_window: int = 1000, verbose = False) -> None:
        
        # inherent from the uniform class
        super(UCB_CD, self).__init__(no_arms, no_output, conf_level, precision_level, stop_type, baseline, reward_type, time_window, verbose)

        # the detection window
        self.trade_off = trade_off

    
    def __str__(self):
        """ -> str"""
        return self.__class__.__name__ 


    def nextpull(self)-> List[int]: 
        '''
        decide which arm(s) to pull next, return a list contains the index of arms
        ''' 
        # UCB sampling with the uncertainty inflation (when trade-off ==1, it recovers UCB)
        newarm_idx =  [self.arms[i]['mu_hat'] + self.trade_off*(self.arms[i]['ucb']-self.arms[i]['mu_hat']) for i in range(self.no_arms)]
        self.next_index  = argsort(newarm_idx)[::-1][:self.no_output]    
           
        return list(self.next_index)                 
