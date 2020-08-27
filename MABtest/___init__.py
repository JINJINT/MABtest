# -*- coding: utf-8 -*-
""" Algorithms for conducting online experiments with adaptive sampling using MAB:
all algorthms have the same interface, as described in :class:'uniform`,
in order to use them in any experiment with the following approach: ::
    my_alg = algo(no_arms)
    my_alg.start()  # start the game
    for t in range(T):
    	if not my_alg.should_stop():  # if have not reach significance
            chosen_arm_t = k_t = my_policy.nextpull()  # chose one arm
            reward_t     = sampled from an arm k_t   # sample a reward
            my_alg.update(k_t, reward_t)       # give it the the policy
"""

# #==== algorithms in stationary environments =======#
# naive random/uniform sampling
from .uniform import uniform
# adaptive sampling with trade-off parameter
from .UCB import UCB
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
from .runsingle import *

from .main import *

__all__ = ["uniform", "uniform_CD", 'uniform_decay', 'UCB', 'UCB_CD', 'UCB_decay', 'TS', 'TS_CD', 'TS_decay', 'confidence', 'estimation','toimport','runsingle', 'main']    

