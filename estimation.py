import math
from numpy import sqrt, log, exp, mean, cumsum, zeros, argsort, argmin, argmax, array
import numpy


def posterior(arm, reward):
	# compute the posterior based on known prior format
	arm['a'] += reward
	arm['b'] += (1-reward)

	return arm['a'], arm['b']


def emp_posterior(arm):
	# compute the posterior based on unknown prior
	pass



def sum(arm, timevar):
	pass



def mean(arm):
	pass



def quantile(arm):
	pass














