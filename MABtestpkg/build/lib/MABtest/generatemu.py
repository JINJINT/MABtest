# scrip fot generate the true mean of the arms according to reward type and time variation
from numpy import sqrt, log, exp, mean, cumsum, zeros, argsort, argmin, argmax, array, random, argwhere, sin, mod, power
from numpy.random import rand
from toimport import *
from plotting import *
import dill
import functools

def generate_mutime(record_max = 10**4, no_arms = 5, mu_type = 'biggap', is_timevar = False, timevar_type = 'General', reward_type = 'Bernoulli', mu_list = None, plot = True, direc = None):
    ''' 
    generate the value of each arm mean as a function of time

    ''' 
    # if not provided the list of mean of arms to start with, then generate it
    if mu_list is None:   
        mu_list = generate_mulist(no_arms, mu_type, timevar_type, reward_type)
    
    # make sure the senatity
    no_arms = len(mu_list)
    
    if is_timevar:  
        # no changes  
        mu_time_list = lambda t, i: mu_list[i]
    else:      
        if timevar_type == 'discrete':
            # continuous changes 
            if mu_type == "biggap":
                f_t = lambda t: 0.1*pow(-1,np.floor(t/50000))*(t>=50000*np.floor(t/50000) and t<50000*(np.floor(t/50000)+1))           
            elif mu_type == "smallgap": 
                f_t = lambda t: 0.01*pow(-1,np.floor(t/500000))*(t>=500000*np.floor(t/500000) and t<500000*(np.floor(t/500000)+1)) 
            else:   
                f_t = lambda t: rand()*pow(-1,np.floor(t/500000))*(t>=500000*np.floor(t/500000) and t<500000*(np.floor(t/500000)+1))  
            mu_time_list = lambda t, i:  mu_list[mod(i+ np.int(np.divide(t,50000)), no_arms)] + f_t(t)

        if timevar_type == 'continuous':
            # continuous changes 
            f_t = lambda t: sin(np.pi*t/500000)+1
            mu_time_list = lambda t, i: mu_list[i]*f_t(t+500000*i)  

        if timevar_type =='realdiscrete':
            # Example from the Yahoo! dataset, from article "Nearly Optimal Adaptive Procedure with Change Detection for Piecewise-Stationary Bandit" (M-UCB) https://arxiv.org/abs/1802.03692
            # 6 arms, 9 discrete change
            mu_list = [[0.071, 0.041, 0.032, 0.030, 0.020, 0.011], 
                        [0.055, 0.053, 0.032, 0.030, 0.008, 0.011], 
                        [0.040, 0.063, 0.032, 0.030, 0.008, 0.011], 
                        [0.040, 0.042, 0.043, 0.030, 0.008, 0.011], 
                        [0.030, 0.032, 0.055, 0.030, 0.008, 0.011], 
                        [0.030, 0.032, 0.020, 0.030, 0.008, 0.021], 
                        [0.020, 0.022, 0.020, 0.045, 0.008, 0.021],  
                        [0.020, 0.022, 0.020, 0.057, 0.008, 0.011], 
                        [0.020, 0.022, 0.034, 0.057, 0.022, 0.011]]   

            mu_time_list = lambda t, i: mu_list[int(np.floor(t/50000))][i]
    
    # save the data to direc
    if direc is not None: 
        if not os.path.exists("%s/%s.pkl"%(direc, filename)):
            filename = 'noarms%d_mu_type%s_timevar_type%s_reward_type%s' % (no_arms, mu_type, timevar_type, reward_type)
            savelambda(mu_time_list, direc, filename)  
        else:
            with open('%s/%s.pkl'%(direc, filename), 'rb') as input:
                mu_dat = dill.load(input) 
            mu_time_list = mu_dat
    
        plotfilename = 'mu_time_list%s'%(filename)
        plotdirec = direc + '/plots'

        if not os.path.exists("%s/%s.pdf"%(plotdirec, plotfilename)) and plot:
    
    # return the list, which is a list of funtion, each represent the mean of arms as a function of time
    return mu_time_list  
    
    

def generate_mulist(no_arms, mu_type, timevar_type, reward_type):
    ''' 
    generate the value of each arm mean

    '''
    if 'sparse' in mu_type:
            signal = np.int(np.ceil(np.power(no_arms, 0.5)))
    elif 'dense' in mu_type:
        signal = np.int(np.ceil(0.5*no_arms))
    if 'strong' in mu_type: 
        sig_a = 10
        sig_mu = 2
    elif 'weak' in mu_type:
        sig_a = 3
        sig_mu = 0.5 

    if mu_type == "biggap":
            mu_list = np.arange(0.1, 0.6, 0.1)
    elif mu_type == 'smallgap':
            mu_list = np.arange(0.1, 0.2, 0.01)
    elif mu_type == 'rand':
        if reward_type == 'Bernoulli':
            mu_list = rand(no_arms)
        elif reward_type == 'Gaussian': 
            mu_list = abs(normal(loc = 0, scale = 1, size = no_arms))
    else:
        if reward_type == 'Bernoulli':
           mu_list = [beta(a = sig_a, b = 1) for i in range(signal)]
           mu_list.extend([beta(a = 1, b = 1) for i in range(no_arms - signal)])
        elif reward_type == 'Gaussian':
           mu_list = [normal(loc = sig_mu, scale = 1) for i in range(signal)]
           mu_list.extend([normal(loc = 0, scale = 1) for i in range(no_arms - signal)])  
        
    mu_list = list(mu_list)
    return mu_list


def plot_mu(plotdirec, plotfilename, no_arms, record_max, mu_time_list, mu_type):
    
    fig, ax = plt.subplots(figsize = (5,5)) 

    # set up label colors
    if len(color_list) < no_arms:
        plot_col = randcolor(no_arms)
    else:
        plot_col = color_list
    
    # set up labels
    labels = ['arm %d' % (i) for i in range(no_arms)]     

    for i in range(no_arms):
        ax.plot(range(record_max), list(map(functools.partial(mu_time_list, i = i), range(record_max))), color = plot_col[i], label = labels[i], lw = 2) 

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))                           
    ax.set_xlabel('Time', labelpad = 20)
    ax.set_ylabel('True mean', labelpad = 20)
    ax.set_ylim((0,1))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    ax.legend(ncol = 1, prop={'size': 10})

    saveplot(plotdirec, plotfilename, ax) 
    return None   
  


