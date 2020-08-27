import pytest
from .uniform import uniform
from .UCB import UCB
from .TS import TS
from .uniform_decay import uniform_decay
from .UCB_decay import UCB_decay
from .TS_decay import TS_decay
from .uniform_CD import uniform_CD
from .UCB_CD import UCB_CD


def test_generatemu():
    pass



def test_singlerun_notimevar():

    mu_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    mu_time_list = generate_mutime(is_timevar = False, mu_list)
    baseline_time = lambda t: baseline

    #====== best arm identification =======#
    # use uniform algo
    mab_uniform = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'uniform',
                 record_time:  lambda t: 10*t, record_max: int = 10**4)

    # use UCB algo
    mab_ucb = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'UCB',
                 record_time:  lambda t: 10*t, record_max: int = 10**4)
    mab_ucb_trade = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'UCB', trade_off = 5,
                 record_time:  lambda t: 10*t, record_max: int = 10**4)

    # use TS algo
    mab_ts = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'TS',
                 record_time:  lambda t: 10*t, record_max: int = 10**4)
    mab_ts_trade = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'TS', trade_off = 5,
             record_time:  lambda t: 10*t, record_max: int = 10**4)

    #====== best arm identification with control =======#

        # use uniform algo
    mab_uniform = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'uniform', stop_type = 'Best-Base', baseline = baseline,
                 record_time:  lambda t: 10*t, record_max: int = 10**4)

    # use UCB algo
    mab_ucb = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'UCB',stop_type = 'Best-Base', baseline = baseline,
                 record_time:  lambda t: 10*t, record_max: int = 10**4)
    mab_ucb_trade = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'UCB',stop_type = 'Best-Base', baseline = baseline, trade_off = 5,
                 record_time:  lambda t: 10*t, record_max: int = 10**4)

    # use TS algo
    mab_ts = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'TS',stop_type = 'Best-Base', baseline = baseline,
                 record_time:  lambda t: 10*t, record_max: int = 10**4)
    mab_ts_trade = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'TS', stop_type = 'Best-Base', baseline = baseline,trade_off = 5,
             record_time:  lambda t: 10*t, record_max: int = 10**4)



    #====== effctive arm identification with FWER control =======#
    # use uniform algo
    mab_uniform = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'uniform',
                 record_time:  lambda t: 10*t, record_max: int = 10**4)

    # use UCB algo
    mab_ucb = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'UCB', stop_type = 'FWER-Base', 
                 record_time:  lambda t: 10*t, record_max: int = 10**4)
    mab_ucb_trade = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'UCB', stop_type = 'FWER-Base', trade_off = 5,
                 record_time:  lambda t: 10*t, record_max: int = 10**4)

    # use TS algo
    mab_ts = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'TS', stop_type = 'FWER-Base',
                 record_time:  lambda t: 10*t, record_max: int = 10**4)
    mab_ts_trade = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'TS', stop_type = 'FWER-Base', trade_off = 5,
             record_time:  lambda t: 10*t, record_max: int = 10**4)


    #====== effctive arm identification with FDR control =======#
    # use uniform algo
    mab_uniform = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'uniform', stop_type = 'FDR-Base',
                 record_time:  lambda t: 10*t, record_max: int = 10**4)

    # use UCB algo
    mab_ucb = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'UCB', stop_type = 'FDR-Base',
                 record_time:  lambda t: 10*t, record_max: int = 10**4)
    mab_ucb_trade = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'UCB', stop_type = 'FDR-Base',trade_off = 5,
                 record_time:  lambda t: 10*t, record_max: int = 10**4)

    # use TS algo
    mab_ts = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'TS', stop_type = 'FDR-Base',
                 record_time:  lambda t: 10*t, record_max: int = 10**4)
    mab_ts_trade = singlerun(mu_list = mu_time_list, no_arms= 5, no_output= 1, samp_type = 'TS', stop_type = 'FDR-Base',trade_off = 5,
             record_time:  lambda t: 10*t, record_max: int = 10**4)



def test_singlerun_timevar_general():
    pass



def test_singlerun_timevar_abrupt():
    pass


