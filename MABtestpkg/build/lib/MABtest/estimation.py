import math
from numpy import sqrt, log, exp, mean, cumsum, zeros, argsort, argmin, argmax, array, power
import numpy
from numpy.random import beta, normal, gamma


def update_posterior(prior_para_1, prior_para_2, reward, reward_type, time_decay = 0):
    if time_decay == 0:
        # compute the posterior based on known prior format
        if reward_type =='Bernoulli': # Bernoulli
            # using Beta Proir Beta(a,b)
            prior_para_1 += reward
            prior_para_2 += (1-reward)
        
        if reward_type == 'Gaussian': # Gaussian with known variance 1 and unknown mean
            # using Gauss prior N(a,b)
            prior_para_1 = (1/(1+prior_para_2))*prior_para_1 + (prior_para_2/(1+prior_para_2))*reward
            prior_para_2 = 1/(1/prior_para_2 + 1) 
        
        if reward_type == 'Poisson': # Poisson with unkonwn mean
            # using gamma prior Gamma(a,b)
            prior_para_1 += reward
            prior_para_2 += 1 
    else:
        # compute the posterior based on known prior format
        if reward_type =='Bernoulli':
            # using Beta Proir Beta(a,b)
            prior_para_1 = (1-time_decay) * prior_para_1 + reward
            prior_para_2 = (1-time_decay) * prior_para_2 + (1-reward)
        
        if reward_type == 'Gaussian': # Gaussian with known precision 1 [precision = (1/sigma^2)] and unknown mean
            # using Gauss prior N(a,1/b)
            prior_para_1 = (time_decay/(time_decay+(1-time_decay)*prior_para_2))*prior_para_1 + (((1-time_decay)*prior_para_2)/(time_decay+(1-time_decay)*prior_para_2))*reward
            prior_para_2 = 1/((1-time_decay)/prior_para_2 + time_decay) 
        
        if reward_type == 'Poisson':
            # using gamma prior Gamma(a,b)
            prior_para_1 += reward
            prior_para_2 += 1 
                   
    return prior_para_1, prior_para_2


def sample_posterior(prior_para_1, prior_para_2, reward_type, var_trade_off = 1):
        # compute the posterior based on known prior format
        # var_trade_off are used to inflate the variance
    no_arms = len(prior_para_1)    
    if reward_type =='Bernoulli':
        # using Beta Proir Beta(a,b)
        return [beta(prior_para_1[i]/power(var_trade_off,2), prior_para_2[i]/power(var_trade_off,2)) for i in range(no_arms)]
    
    if reward_type == 'Gaussian': # Gaussian with known precision 1 [precision = (1/sigma^2)] and unknown mean
        # using Gauss prior N(a,1/b)
        return [normal(loc = prior_para_1[i], scale = sqrt(prior_para_2[i])*var_trade_off) for i in range(no_arms)]
    
    if reward_type == 'Poisson':
        # using gamma prior Gamma(a,b)
        return [gamma(prior_para_1[i]/power(var_trade_off,2), prior_para_2[i]/power(var_trade_off,2)) for i in range(no_arms)]



















