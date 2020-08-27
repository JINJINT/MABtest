import numpy as np
import sys

from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate
from numpy.random import randn, rand

import os
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from .toimport import *
from .runsingle  import *
import itertools
import dill
import h5py

## Plotting settings
matplotlib.use('Agg')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize']= 20
mpl.rcParams['ytick.labelsize']= 20
plt.switch_backend('agg')
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)

color_list = ['firebrick', 'gray', 'pink', 'green', 'darkorange', 'royalblue','mediumslateblue']


#===============================================
# 
# plot over many trials
#
#===============================================    

def plot_MAB(mabdat, direc, filename, bound =True, arms_idx = None):
    # mabdat: a list of objects from class single_run
    # direc: the directory to save the plots
    # arms_idx: which arms to plot

    record_times = mabdat.record_time_list
    direc = direc + '/plots'

    #=========== plot mean

    if arms_idx is None:
        arms_idx = list(range(mabdat.no_arms))

    if len(color_list) < len(arms_idx):
        plot_col = randcolor(len(arms_idx))
    else:
        plot_col = color_list    
    if mabdat.time_decay == 'none':
        labels = ['arm %d, $\mu$ = %.5f' % (idx, mabdat.mu_list[idx]) for idx in arms_idx]   
        no_outputidx = mabdat.bestidx
        for i, idx in enumerate(arms_idx):
            if idx in no_outputidx:
                labels[i] += '(top%d)' % (mabdat.no_output)
    else:
        labels = ['arm %d' % (idx) for idx in arms_idx]            
    
    fig, ax = plt.subplots(figsize = (5,5)) 

    min_y = np.inf
        
    for i, idx in enumerate(arms_idx):
        lcb_list = mabdat.allarms[idx]['lcb']
        ucb_list = mabdat.allarms[idx]['ucb']
        mu_list = mabdat.allarms[idx]['mu_hat']
        min_y_temp = min(mu_list)
        if min_y > min_y_temp:
            min_y = min_y_temp
        if idx == mabdat.bestidx[0]: 
            max_y = max(ucb_list)
        color = plot_col[i]   
        ax.errorbar(record_times, mu_list, color = color, linestyle = '-', lw= 1, label=labels[i])
        if bound: 
            ax.errorbar(record_times, lcb_list, color = color, linestyle = '-.', lw= 0.5)
            ax.errorbar(record_times, ucb_list, color = color, linestyle = '-.', lw= 0.5)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                ncol = 1, prop={'size': 10})
                       
    ax.set_xlabel('Number of pulls', labelpad=7)
    ax.set_ylabel('Estimated mean', labelpad=7)
    ax.set_xlim((record_times[0], record_times[-1]))
    ax.set_ylim((0,1))

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))    

    # specially mark the stopping time
    if mabdat.auto_stop and mabdat.should_stop:
        ax.annotate('Stop', 
            xy=(mabdat.should_stop_time, 0), 
            xytext=(mabdat.should_stop_time, -0.1), 
            arrowprops = dict(facecolor='black', shrink=0.05))

    meanfilename = filename + '_mean'
    if bound:
        meanfilename += 'withbound'
    saveplot(direc, meanfilename, ax)
    

    # ====== plot reward

    fig, ax = plt.subplots(figsize = (5,5)) 

    reward_list = mabdat.accreward_list
  
    ax.errorbar(record_times, reward_list, marker = '.', linestyle = '-.', lw= 2, markersize = 4)

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))                           
    ax.set_xlabel('Number of pulls', labelpad=7)
    ax.set_ylabel('Reward', labelpad=7)
    ax.set_xlim((record_times[0], record_times[-1]))

    # specially mark the stopping time
    if mabdat.auto_stop and mabdat.should_stop:
        ax.annotate('Stop', 
            xy=(mabdat.should_stop_time, -5), 
            xytext=(mabdat.should_stop_time, -5.5), 
            arrowprops = dict(facecolor='black', shrink=0.05))

    rwfilename = filename + '_reward'
    saveplot(direc, rwfilename, ax)



    #======== plot regret

    fig, ax = plt.subplots(figsize = (5,5)) 

    regret_list = list(map(operator.sub, mabdat.accreward_optimal_list, mabdat.accreward_list))
  
    ax.errorbar(record_times, regret_list, marker = '.', linestyle = '-.', lw= 2, markersize = 4)

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))                           
    ax.set_xlabel('Number of pulls', labelpad=7)
    ax.set_ylabel('Regret', labelpad=7)
    ax.set_xlim((record_times[0], record_times[-1]))

    # specially mark the stopping time
    if mabdat.auto_stop and mabdat.should_stop:
        ax.annotate('Stop', 
            xy=(mabdat.should_stop_time, -5), 
            xytext=(mabdat.should_stop_time, -5.5), 
            arrowprops = dict(facecolor='black', shrink=0.05))

    rgfilename = filename + '_regret'
    saveplot(direc, rgfilename, ax)


#===============================================
# 
# plot over many trials
#
#===============================================       

def plot_MAB_real(mabdat, direc, filename, bound =True, arms_idx = None):
    # mabdat: a list of objects from class single_run
    # direc: the directory to save the plots
    # arms_idx: which arms to plot

    record_times = mabdat.record_time_list
    direc = direc + '/plots'

    #=========== plot mean

    if arms_idx is None:
        arms_idx = list(range(mabdat.no_arms))

    if len(color_list) < len(arms_idx):
        plot_col = randcolor(len(arms_idx))
    else:
        plot_col = color_list    
    if mabdat.time_decay == 'none':
        labels = ['arm %d, $\mu$ = %.5f' % (idx, mabdat.mu_list[idx]) for idx in arms_idx]   
        no_outputidx = mabdat.bestidx
        for i, idx in enumerate(arms_idx):
            if idx in no_outputidx:
                labels[i] += '(top%d)' % (mabdat.no_output)
    else:
        labels = ['arm %d' % (idx) for idx in arms_idx]            
    
    fig, ax = plt.subplots(figsize = (5,5)) 

    min_y = np.inf

    ttbest = [0,4,3,2]
    realmu_list = [[0.6,0.5,0.4,0.3,0.2],
                   [0.3,0.2,0.1,0.0,0.4],
                   [0.4,0.3,0.2,0.6,0.5],
                   [0.1,0.0,0.4,0.3,0.2]]
    
    # ttbest = [0,0,1,2,2,1,3,3,3]
    # realmu_list = [[0.071, 0.041, 0.032, 0.030, 0.020, 0.011], 
    #                     [0.055, 0.053, 0.032, 0.030, 0.008, 0.011], 
    #                     [0.040, 0.063, 0.032, 0.030, 0.008, 0.011], 
    #                     [0.040, 0.042, 0.043, 0.030, 0.008, 0.011], 
    #                     [0.030, 0.032, 0.055, 0.030, 0.008, 0.011], 
    #                     [0.030, 0.032, 0.020, 0.030, 0.008, 0.021], 
    #                     [0.020, 0.022, 0.020, 0.045, 0.008, 0.021],  
    #                     [0.020, 0.022, 0.020, 0.057, 0.008, 0.011], 
    #                     [0.020, 0.022, 0.034, 0.057, 0.022, 0.011]] 

    for tt, bestidx in enumerate(ttbest):
        begin = tt*50
        end = min((tt+1)*50, np.int(record_times[-1]/1000))
        mu_list_true = [realmu_list[tt][bestidx] for i in range(begin,end)]
        mu_list_hat = mabdat.allarms[bestidx]['mu_hat'][begin:end]
        lcb_list = mabdat.allarms[bestidx]['lcb'][begin:end]
        ucb_list = mabdat.allarms[bestidx]['ucb'][begin:end]
        color = plot_col[bestidx]         
        ax.plot(record_times[begin:end], mu_list_hat, color = color, linestyle = '-', lw= 1)
        ax.plot(record_times[begin:end], mu_list_true, color = color, linestyle = ':', lw = 3)
        ax.plot(record_times[begin:end], lcb_list, color = color, linestyle = '-.', lw= 0.7)
        ax.plot(record_times[begin:end], ucb_list, color = color, linestyle = '-.', lw= 0.7)
        plt.axvline(x=record_times[end-1], color = 'black', linestyle = '-.', lw= 1)
                       
    ax.set_xlabel('Number of pulls', labelpad=20)
    ax.set_ylabel('Estimated mean and confidence bounds', labelpad=20)
    ax.set_xlim((record_times[0], record_times[-1]))
    ax.set_ylim((0,1))

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))    

    meanfilename = filename + '_mean'
    if bound:
        meanfilename += 'withbound'
    saveplot(direc, meanfilename, ax)


def plot_regret(direc, no_arms, no_output, trade_off , conf_level, precision_level, samp_type, 
        reward_type, stop_type, record_break, record_max, baseline, timevar_type, time_decay, real_methodlist):

    fig, ax = plt.subplots(figsize = (5,5)) 
    #fig.suptitle('Experiments across different $\\trade_off$')
    
    # set up label colors

    plot_col = ['black', 'purple', 'royalblue', 'brown', 'darkorange', 'green', 'red', 'olive', 'pink']

    stoptime_mat = []    
    regretstop_mat = []
    regret_end = []

    labels = real_methodlist   

    # collecting data and average over trials
    for i, real_method in enumerate(real_methodlist):

        if real_method =='TS-Decay':
            time_decay = 0.0001
        else:
            time_decay = 0.
        color = plot_col[i] 
        
        # define the file name as incoporating the  parameter settings
        filename = 'MAB_noarms%d_nooutput%d_%s_%s_%s_trade_off%.2f_conf_level%.2f_precision_level%.2f_break%d_max%d' %\
                    (no_arms, no_output, samp_type, reward_type, stop_type, trade_off, conf_level, precision_level, record_break, record_max)  

        if 'Base' in stop_type:   
            filename += '_baseline%4f'%(baseline) 

        if not is_timevar:   
            filename += '_timevar%s'%(timevar_type) 
        
        all_filenames = [filename for filename in os.listdir(direc) if filename.startswith(filename_pre)]

        if all_filenames == []:
            print("No file found start with %s!" % (filename_pre))
            sys.exit()

        sum_regret_list = 0.

        for filename in all_filenames:
            filenam = '%s/%s'%(direc, filename)
            with open(filenam, 'rb') as input:
                mabdat = dill.load(input)
            plot_MAB_real(mabdat, direc, filename)    
            # gather the reward info
            reward_list = mabdat.accreward_list
            regret_list = list(map(operator.sub, mabdat.accreward_optimal_list, mabdat.accreward_list))
            regret_end.append(regret_list[::-1][0])
            record_times = mabdat.record_time_list
            sum_regret_list = np.add(sum_regret_list, regret_list)
            #ax.plot(record_times, regret_list, lw= 0.6, color = color, alpha = 0.15)

        mean_regret_list = np.divide(sum_regret_list, len(all_filenames))
        ax.plot(record_times, mean_regret_list, lw= 2, color = color, label = labels[i])    
        
    f = h5py.File(direc+'others.hdf5', 'r')
    g = f['env_0']
    dat = g['cumulatedRegret']
    mean_regret_list = dat[0]
    ax.plot(range(len(mean_regret_list)), mean_regret_list, lw= 2, color = plot_col[4], label = 'UCB')
    mean_regret_list = dat[5]
    ax.plot(range(len(mean_regret_list)), mean_regret_list, lw= 2, color = plot_col[5], label = 'SW-UCB')
    mean_regret_list = dat[6]
    ax.plot(range(len(mean_regret_list)), mean_regret_list, lw= 2, color = plot_col[6], label = 'M-UCB')
    mean_regret_list = dat[11]
    ax.plot(range(len(mean_regret_list)), mean_regret_list, lw= 2, color = plot_col[7], label = 'GLR-UCB')
    mean_regret_list = dat[2]
    ax.plot(range(len(mean_regret_list)), mean_regret_list, lw= 2, color = plot_col[8], label = 'Exp3.S')

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_xlabel('Number of pulls', labelpad=20)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_ylabel('Regret', labelpad=20)

    plotfilename = 'avg_regret_real'
    plotdirec = direc + '/plots'
    saveplot(plotdirec, plotfilename, ax)  


def plot_summary(direc, no_arms, no_output, trade_offlist , conf_level, precision_level, samp_type, 
        reward_type, stop_type, record_break, record_max, baseline, timevar_type, time_decay):

    fig, ((ax1,ax2,ax5),(ax3,ax4,ax6)) = plt.subplots(figsize = (12,8), dpi=100, ncols=3, nrows = 2) 
    #fig.suptitle('Experiments across different $\\trade_off$')
    
    # set up label colors
    if len(color_list) < len(trade_offlist):
        plot_col = randcolor(len(trade_offlist))
    else:
        plot_col = color_list

    #plot_col[trade_offlist.index(1)] = 'black'    

    # set up labels
    if samp_type=='TTTS':
        labels = ['$\\beta$= %.2f' % (trade_off) for trade_off in trade_offlist] 
        if 1. in trade_offlist:
            idx = trade_offlist.index(1.)
            labels[idx] += ' (TS)'

    if samp_type == 'UCB':
        labels = ['$\\alpha$= %.2f' % (trade_off) for trade_off in trade_offlist] 
        if 1. in trade_offlist:
            idx = trade_offlist.index(1.)
            labels[idx] += ' (UCB)'

    if samp_type=='TS':
        labels = ['$\\alpha$= %.2f' % (trade_off) for trade_off in trade_offlist] 
        if 1. in trade_offlist:
            idx = trade_offlist.index(1.)
            labels[idx] += ' (TS)'    

    if samp_type=='Adversarial':
        labels = ['$\\gamma$= %.2f' % (trade_off) for trade_off in trade_offlist] 
        if 0. in trade_offlist:
            idx = trade_offlist.index(0.)
            labels[idx] += ' (Softmax)'          
                
    if -1. in trade_offlist:
        uniform_idx = trade_offlist.index(-1.)
        labels[uniform_idx] += ' (uniform)'    

    stoptime_mat = []    
    regretstop_mat = []

    # collecting data and average over trials
    for i, trade_off in enumerate(trade_offlist):

        color = plot_col[i]  

        if trade_off == -1:
            samp_typ = 'uniform'
        else:
            samp_typ = samp_type        
        
        # define the file name as incoporating the  parameter settings
        filename = 'MAB_noarms%d_nooutput%d_%s_%s_%s_trade_off%.2f_conf_level%.2f_precision_level%.2f_break%d_max%d' %\
                    (no_arms, no_output, samp_typ, reward_type, stop_type, trade_off, conf_level, precision_level, record_break, record_max)  

        if 'Base' in stop_type:   
            filename += '_baseline%4f'%(baseline) 

        if not is_timevar:   
            filename += '_timevar%s'%(timevar_type) 
       
        all_filenames = [filename for filename in os.listdir(direc) if filename.startswith(filename_pre)]

        if all_filenames == []:
            print("No file found start with %s!" % (filename_pre))
            sys.exit()

        sum_reward_list = 0.
        sum_regret_list = 0.
        stoptime = []
        regretstop = []

        for filename in all_filenames:
            filenam = '%s/%s'%(direc, filename)
            print(filenam)
            with open(filenam, 'rb') as input:
                mabdat = dill.load(input)

            # gather the reward info
            reward_list = mabdat.accreward_list
            regret_list = list(map(operator.sub, mabdat.accreward_optimal_list, mabdat.accreward_list))
            record_times = mabdat.record_time_list
            sum_reward_list = np.add(sum_reward_list, reward_list)
            sum_regret_list = np.add(sum_regret_list, regret_list)
            #ax1.plot(record_times, regret_list, lw= 0.6, color = color, alpha = 0.35)
            ax2.plot(record_times, reward_list, lw= 0.6, color = color, alpha = 0.35)
 
            # gather the stop time info
            stoptime.append(mabdat.should_stop_time)
            # gather the regret at stop
            if mabdat.accregret_stop is None:
                mabdat.accregret_stop = regret_list[::-1][0]
            regretstop.append(mabdat.accregret_stop)
        
        mean_reward_list = np.divide(sum_reward_list, len(all_filenames))
        mean_regret_list = np.divide(sum_regret_list, len(all_filenames))

        ax1.plot(record_times, mean_regret_list, lw= 1.2, color = color)
        ax2.plot(record_times, mean_reward_list, lw= 1.2, color = color, label = labels[i])
        
        stoptime = np.array(stoptime)
        stoptime[stoptime<0] = record_times[::-1][0]+1
        stoptime_mat.append(stoptime) 
        stopprob_list = [np.mean(stoptime<=t) for t in record_times]
        ax3.plot(record_times, stopprob_list, lw= 1.2, color = color)

        regretstop_mat.append(regretstop)  

    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1.set_xlabel('Number of pulls', labelpad=12)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.set_ylabel('Regret', labelpad=12)
    #ax1.set_ylim((0,5*10**4))

    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax2.set_xlabel('Number of pulls', labelpad=12)    
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.set_ylabel('Reward', labelpad=12)

    # stopping probability
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax3.set_xlabel('Number of pulls', labelpad=12)
    ax3.set_xlim((0,2*10**5))
    ax3.set_ylabel('Stopping probability', labelpad=12)

    # box plot of stopping time
    box = ax4.boxplot(stoptime_mat, patch_artist=True, labels = trade_offlist)
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax4.set_ylabel('Stopping time', labelpad=12)
    ax4.set_xlabel('trade off', labelpad=12)
    ax4.set_ylim((0,2*10**5))
    #ax4.hlines(y=mean(stoptime_mat[0]), xmin=0, xmax=6, linewidth=2, color='r')
    # for patch, color in zip(box['boxes'], plot_col):
    #     patch.set_facecolor(color)  

    # box plot of reward at stopping  
    box = ax6.boxplot(regretstop_mat, patch_artist=True, labels = trade_offlist)
    ax6.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax6.set_ylabel('Regret at stopp', labelpad=12)
    ax6.set_xlabel('trade_off', labelpad=12)
    #ax6.hlines(y=mean(regretstop_mat[0]), xmin=0, xmax=6, linewidth=2, color='r')
    # for patch, color in zip(box['boxes'], plot_col):
    #     patch.set_facecolor(color) 
        

    ax5.axis('off')     

    lgd = fig.legend(loc='upper right', prop= {'size': 15})
    fig.tight_layout()

    plotfilename = 'avg_results_trade_off_all%s'%(samp_type)
    plotdirec = direc + '/plots'
    saveplot(plotdirec, plotfilename, lgd)  





def plot_pulls(direc, no_arms, no_output, trade_offlist , conf_level, precision_level, samp_type, 
        reward_type, stop_type, record_break, record_max, baseline, timevar_type, time_decaylist, arms_idx = None):
    if arms_idx is None:
        arms_idx = list(range(no_arms))  

    # set up label colors
    if len(color_list) < len(arms_idx) + 1:
        plot_col = randcolor(len(arms_idx)+1)
    else:
        plot_col = color_list
  
    mu_list = None

    # collecting data and average over trials
    for trade_off, time_decay in product(trade_offlist, time_decaylist):  
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if trade_off == -1:
            samp_typ = 'uniform'
        else:
            samp_typ = samp_type  

        # define the file name as incoporating the  parameter settings
        filename = 'MAB_noarms%d_nooutput%d_%s_%s_%s_trade_off%.2f_conf_level%.2f_precision_level%.2f_break%d_max%d' %\
                    (no_arms, no_output, samp_typ, reward_type, stop_type, trade_off, conf_level, precision_level, record_break, record_max)  

        if 'Base' in stop_type:   
            filename += '_baseline%4f'%(baseline) 

        if not is_timevar:   
            filename += '_timevar%s'%(timevar_type) 
        
        all_filenames = [filename for filename in os.listdir(direc) if filename.startswith(filename_pre)]

        if all_filenames == []:
            print("No file found start with %s!" % (filename_pre))
            sys.exit()        

        pull_prob_sum = 0.   
        
        for filename in all_filenames:  
            # get the arms that is being pulled
            filenam = '%s/%s'%(direc, filename)
            print(filenam)
            with open(filenam, 'rb') as input:
                mabdat = dill.load(input)
            record_time_list = mabdat.record_time_list
            if timevar_type =='none':
                if mu_list is None:
                    mu_list = mabdat.mu_list
                    # set up labels
                    labels = ['arm %d, $\mu$= %.2f' % (i, mu_list(0,i)) for i in arms_idx]   
                    if 'Control' in stop_type:
                        labels.append('baseline, $\mu$= %.2f' % (baseline))
            else:
                labels = ['arm %d' % (i) for i in arms_idx]   
                            
            pullarm_list = list(itertools.chain.from_iterable(mabdat.pullarm))
            pull_prob = [[] for i in arms_idx]
            record_times = [] 
            record_times_break = []
            temp = [0. for i in range(len(arms_idx))]
            tt = 1
            for t, arm in enumerate(pullarm_list):
                if arm is not None:
                    for i in range(len(arms_idx)):
                        temp[i] += 1.*(arms_idx[i]==arm)
                if t > record_time_list[tt]:
                    for i in range(len(arms_idx)):
                        pull_prob[i].append(temp[i])
                        temp[i] = 0.   
                    record_times.append(record_time_list[tt])    
                    record_times_break.append(record_time_list[tt]-record_time_list[tt-1])
                    tt +=1 
            pull_prob = np.divide(pull_prob, record_times_break)
            for i in range(len(arms_idx)):
                ax.plot(record_times, pull_prob[i], lw= 0.6, color = plot_col[i], alpha = 0.35)
            pull_prob_sum = np.add(pull_prob_sum, pull_prob)  

        pull_prob_mean =  pull_prob_sum/len(all_filenames)
        for i in range(len(arms_idx)):
            ax.plot(record_times, pull_prob_mean[i], lw= 1.2, color = plot_col[i], label = labels[i])

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.set_xlabel('Number of pulls', labelpad=12)
        ax.set_ylabel('Pull probability', labelpad=12)  
        ax.set_ylim((-0.1,1.1))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                ncol = 1, prop={'size': 10})

        plotfilename = 'avg_pulls_trade_off%.2f_%s_%s_time_decay%.4f'%(trade_off, samp_type, timevar_type, time_decay)
        plotdirec = direc + '/plots'
        saveplot(plotdirec, plotfilename, ax)    



def plot_output(direc, no_arms, no_output, trade_offlist , conf_level, precision_level, samp_type, 
        reward_type, stop_type, record_break, record_max, baseline, timevar_type, time_decay):

    fig1, ax1 = plt.subplots(figsize = (8,5), dpi=100)
    #fig2, ax2 = plt.subplots(figsize = (8,5), dpi=100)  

    # set up labels
    if samp_type=='TTTS':
        labels = ['$\\beta$= %.2f' % (trade_off) for trade_off in trade_offlist] 
        if 1. in trade_offlist:
            idx = trade_offlist.index(1.)
            labels[idx] += ' (TS)'

    if samp_type == 'UCB':
        labels = ['$\\alpha$= %.2f' % (trade_off) for trade_off in trade_offlist] 
        if 1. in trade_offlist:
            idx = trade_offlist.index(1.)
            labels[idx] += ' (UCB)'

    if samp_type=='TS':
        labels = ['$\\alpha$= %.2f' % (trade_off) for trade_off in trade_offlist] 
        if 1. in trade_offlist:
            idx = trade_offlist.index(1.)
            labels[idx] += ' (TS)'    

    if samp_type=='Adversarial':
        labels = ['$\\gamma$= %.2f' % (trade_off) for trade_off in trade_offlist] 
        if 0. in trade_offlist:
            idx = trade_offlist.index(0.)
            labels[idx] += ' (Softmax)'          
                
    if -1. in trade_offlist:
        uniform_idx = trade_offlist.index(-1.)
        labels[uniform_idx] += ' (uniform)'         

    # set up label colors
    if len(color_list) < len(trade_offlist) :
        plot_col = randcolor(len(trade_offlist))
    else:
        plot_col = color_list
   

    power_mat = []
    mu_list = None 

    # collecting data and average over trials
    for i, trade_off in enumerate(trade_offlist):

        if trade_off == -1:
            samp_typ = 'uniform'
        else:
            samp_typ = samp_type   

        # define the file name as incoporating the  parameter settings
        filename = 'MAB_noarms%d_nooutput%d_%s_%s_%s_trade_off%.2f_conf_level%.2f_precision_level%.2f_break%d_max%d' %\
                    (no_arms, no_output, samp_typ, reward_type, stop_type, trade_off, conf_level, precision_level, record_break, record_max)  

        if 'Base' in stop_type:   
            filename += '_baseline%4f'%(baseline) 

        if not is_timevar:   
            filename += '_timevar%s'%(timevar_type) 
        
        all_filenames = [filename for filename in os.listdir(direc) if filename.startswith(filename_pre)]

        if all_filenames == []:
            print("No file found start with %s!" % (filename_pre))
            sys.exit() 

        power = []  
        power_time_sum = 0. 

        err = []  
        err_time_sum = 0. 

        for j, filename in enumerate(all_filenames):
            # get the arms that is being pulled
            filename = '%s/%s'%(direc, filename)
            with open(filename, 'rb') as input:
                mabdat = dill.load(input)     
                output_idx = mabdat.output
                output_stop = None
                if mu_list is None:
                    mu_list = [mabdat.mu_list(0,i) for i in range(no_arms)]
                for output in output_idx: 
                    if output is not None:
                        output_stop = output
                        break
                if output_stop is None:
                    output_stop = output_idx[-1]        
                optimal_idx = argsort(mu_list)[::-1][:no_output]
                
                if stop_type == 'Best':
                    power.append(len(set(output_stop) & set(optimal_idx))/no_output)

                elif stop_type == 'BestControl':
                    if baseline > max(mu_list) - precision_level: # the null hypothesis is true
                        power.append(1.*(output_stop==[]))
                    else: # the null hypothesis is not true
                        power.append(len(set(output_stop) & set(optimal_idx))/no_output )
                        
                elif stop_type == 'FDRControl':
                    positive_idx = np.concatenate(argwhere(np.array(mu_list)> baseline))
                    power_time = np.array([len(set(output) & set(positive_idx))/len(positive_idx) if output is not None else 0 for output in output_idx ])
                    power_time_sum = np.add(power_time_sum, power_time)

                    nonpositive_idx = np.concatenate(argwhere(np.array(mu_list)<= baseline))
                    err_time = np.array([len(set(output) & set(nonpositive_idx))/max(len(output),1) if output is not None else 0 for output in output_idx ])
                    err_time_sum = np.add(err_time_sum, err_time)    

        if 'FDR' not in stop_type:   
            power_mat.append(power)
        else:
            power_time_mean = power_time_sum/len(all_filenames)
            err_time_mean = err_time_sum/len(all_filenames)
            ax1.plot(range(len(output_idx)), power_time_mean, lw= 1.5, color = plot_col[i], label = labels[i]) 
            #ax2.plot(range(len(output_idx))[:2000], power_time_mean[:2000], lw= 1.5, color = plot_col[i], label = labels[i])       
            #ax2.plot(range(len(output_idx)), err_time_mean, lw= 1.5, color = plot_col[i], label = labels[i])         
    
    plotfilename = 'avg_output_trade_off%s_timevar_type%s'%(samp_type, timevar_type)
    plotdirec = direc + '/plots'
    
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax1.set_ylabel('Power', labelpad=20)
    ax1.set_xlabel('Number of Pulls', labelpad=25)
    saveplot(plotdirec, plotfilename +'long', ax1) 
    
    # ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # ax2.set_ylabel('Power', labelpad=25)
    # ax2.set_xlabel('Number of Pulls', labelpad=25)
    # ax2.set_ylim((0,0.6))
    # saveplot(plotdirec, plotfilename + 'short', ax2) 






        





            






























