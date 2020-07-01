import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import random


'''function related to saving files'''

def saveres(mat, direc, filename, ext = 'dat', verbose = True):
    # save as numpy matrix
    filename = "%s.%s" % (filename, ext)
    if not os.path.exists(direc):
        os.makedirs(direc)
    savepath = os.path.join(direc, filename)
    np.savetxt(savepath, mat, fmt='%.3e', delimiter ='\t')
    if verbose:
        print("Saving results to %s" % savepath)

def saveobject(obj, direc, filename, ext='pkl'):
    # save any object
    filename = "%s.%s" % (filename, ext)
    if not os.path.exists(direc):
        os.makedirs(direc)
    savepath = os.path.join(direc, filename)
    with open(savepath, 'wb') as output:  # Overwrites any existing file.
        print("Saving object to %s" % savepath)
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)        
 

def saveplot(direc, filename, lgd, ext = 'pdf',  close = True, verbose = True):
    filename = "%s.%s" % (filename, ext)
    if not os.path.exists(direc):
        os.makedirs(direc)
    savepath = os.path.join(direc, filename)
    plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    if verbose:
        print("Saving figure to %s" % savepath)
    if close:
        plt.close()

'''function related to other utils'''

def str2list(string, type = 'int'):
    str_arr =  string.split(',')
    if type == 'int':
        str_list = [int(char) for char in str_arr]
    elif type == 'float':
        str_list = [float(char) for char in str_arr]
    return str_list

def list2str(lists):
    string = ''
    for i in lists:
        string = string + str(i)
    return string 

def getCom(seq): 
    combinations = list() 
    for i in range(0,len(seq)): 
        for j in range(0,len(seq)): 
            combinations.append([seq[i],seq[j]]) 
    return combinations


def randcolor(num):
    # generate random colors
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num)]    
    return color  

