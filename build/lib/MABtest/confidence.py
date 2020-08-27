"""A number of confidence bounds for different statistical 
settings (Gaussians and Bernoullis) as a function of time."""
''' 
got the reference from Fanny Yang's paper: https://github.com/fanny-yang/MABFDR
'''
from scipy.stats import norm as normal
from numpy import sqrt, log, exp, cumsum, power
import numpy as np
from numpy.random import randn
import ipdb

def _binary_search(a: float, b: float, f, precision: float = 1e-6)-> float:
    """f(x) returns True if threshold is greater than x, False otherwise."""
    pivot = (a+b)/2.
    while b-a > precision:
        pivot = (a+b)/2.
        if f(pivot):
            b = pivot
        else:
            a = pivot
    return pivot

class ConfidenceBound():
    """Provides class for confidence bound for different statistical settings.

    Intended use:
        mu_hat = mean(mu + randn() for _ in range(t))

        cb = ConfidenceBound('SubGaussian')
        u = cb.upper(mu_hat, conf_level, t, anytime=True)
        l = cb.lower(mu_hat, conf_level, t, anytime=True)

        Then l <= mu <= u with probability at least 1-conf_level. If anytime=True then
        this relation holds for all t simultaneously. 
    """

    def __init__(self, name: str ='SubGaussian', anytime: bool = True):
        """Parameters of confidence bound determined by name.
        All subGaussian tail-bounds are for scale-1 RVs.

        Args:
            anytime: Boolean inidcating that the bound should hold for all time simultaneously.
                If not LIL bound, anytime bound is accomplished by summable 1/(2*t^2) union bound.
            name: what kind of RV
                SubGaussian: typical sub-gaussian tail bound of scale 1
                Gaussian_Exact: tail probability for Gaussian of scale 1
                SubGaussianLIL: SubGaussian tail bound but using LIL union bound instead of 
                    the trivial union bound set by setting anytime=True with SubGaussian bound.
                    When using this bound name, anytime must be set equal to True or raises exception
                BernoulliLIL: rewards Bernoulli, uses SubGaussian-LIL bound 
                Bernoulli: rewards are bernoulli in {0, 1} and uses Chernoff KL bound
        """
        self.name = name
        self.anytime = anytime

        if 'LIL' in self.name:
            assert self.anytime==True, 'When using \'SubGaussianLIL\' you must set anytime==True'

    def _kl(self, a: float, b: float)-> float:
        if a == 0:
            return -log(1.-b)
        elif a == 1:
            return -log(b)
        elif b == 0:
            return float('inf')
        elif b == 1:
            return float('inf')
        else:
            return a*log(a/b) + (1.-a)*log((1.-a)/(1.-b))

    def lower(self, mu_hat: float, conf_level:float, t, sigma: float = 1)-> float:
        if conf_level==0. :
            ipdb.set_trace()
        if self.name in ['BernoulliLIL', 'SubGaussianLIL', 'SubGaussian', 'Gaussian_Exact']:
            return 2*mu_hat-self.upper(mu_hat, conf_level, t, sigma)
        elif self.name == 'Bernoulli':
            conf = conf_level
            if self.anytime: 
                conf = conf_level/(2.*t**2)
            return _binary_search(0., mu_hat, lambda x: t*self._kl(mu_hat, x) < log(1./conf))

    # CURRENTLY ONLY SUPPORTING Gaussians with sigma = 1!
    def upper(self, mu_hat: float, conf_level: float , t, sigma: float = 1)-> float:
        # if 'Bernoulli' in self.name and sigma!=1.:
        #     sigma = 0.5 # SubGaussian parameter
        if sigma == 0: 
            if 'Bernoulli' in self.name:
                sigma = 0.5
            if 'Gaussian' in self.name:
                sigma = 1    
        if 'LIL' in self.name:
            #if self.name =='SubGaussianLIL': (same for Bernoulli LIL)
            temp = log(1./conf_level) + 3*log(log(1./conf_level)) + 1.5*log(log(exp(1)*t))
            return mu_hat + sqrt(2.*pow(sigma,2)*temp/t)
        else:
            conf = conf_level
            if self.anytime: 
                conf = conf_level/(2.*t**2)
            if self.name == 'SubGaussian':
                return mu_hat + sqrt(2.*log(1./conf)/t)
            elif self.name == 'Gaussian_Exact':
                return mu_hat + _binary_search(0., sqrt(2.*log(1./conf)), lambda x: 1-normal.cdf(x) < conf)/sqrt(t)
            elif self.name == 'Bernoulli':
                return _binary_search(mu_hat, 1., lambda x: t*self._kl(mu_hat, x) > log(1./conf))

    def lower_timevar(self, mu_hat: float, conf_level: float, t, time_decay, total_t)->float:
        if conf_level==0.:
            ipdb.set_trace()
        if total_t==0 or t==0:
            return -np.inf      
        return 2*mu_hat-self.upper_timevar(mu_hat, conf_level, t, time_decay, total_t)

    def upper_timevar(self, mu_hat: float, conf_level: float, t, time_decay, total_t: int)->float: 
        if conf_level==0.:
            ipdb.set_trace()
        if 'Bernoulli' in self.name:
            bound = 1 # 1 by definition
        elif 'Gaussian' in self.name:
            bound = 1  # 1 for now, could be specify later      
        if total_t==0 or t==0:
            return np.inf   
        else:     
            return mu_hat + 2*bound*sqrt(log(1/time_decay-power(1-time_decay, total_t))/t) # from dicounted UCB, this bound is only valid when totoal_t is far away from the change point
          

   





