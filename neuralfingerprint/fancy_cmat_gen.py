'''
Labelled cmat display borrowed from Landrum 2014 paper
'''

import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pickle as pkl
from collections import defaultdict

def labelled_cmat(cmat,labels,#figsize=(20,15),
    labelExtras=None, #dpi=300, 
    xlabel=True, ylabel=True, rotation=90):
    
    #rowCounts = np.array(sum(cmat,1),dtype=float)
    #cmat_percent = cmat/rowCounts[:,None]
    #zero all elements that are less than 1% of the row contents
    #ncm = cmat_percent*(cmat_percent>threshold)

    #fig = figure(1,figsize=figsize,dpi=dpi)
    fig = figure(1)
    ax = fig.add_subplot(1,1,1)
    #fig.set_size_inches(figsize)
    #fig.set_dpi(dpi)
    pax=ax.pcolor(cmat,cmap=plt.get_cmap('Blues',100000),vmin=0, vmax=1)#cm.ocean_r)
    ax.set_frame_on(True)

    #print cmat.shape[0]
    print np.sum(cmat)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(cmat.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(cmat.shape[1])+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if labelExtras is not None:
        labels = [' %s %s'%(x,labelExtras[x].strip()) for x in labels]
    
    ax.set_xticklabels([], minor=False) 
    ax.set_yticklabels([], minor=False)

    if xlabel:
        ax.set_xticklabels(labels, minor=False, rotation=rotation, horizontalalignment='left',size=14) 
        plt.xlabel('True reaction label')
        ax.xaxis.set_label_position('top')
        #ax.set_xticklabels(labels, minor=False, rotation=0, horizontalalignment='left',size=14) 
    if ylabel:
        ax.set_yticklabels(labels, minor=False,size=14)
        plt.ylabel('Predicted reaction label')

    #set(gca,'fontsize',8)

    #ax.grid(True)
    fig.colorbar(pax)
    axis('tight')
    plt.show()

if __name__ == '__main__':
    results = pkl.load(open('opt/class_3_3_perbenzacid_morgan1_50eps_results.dat'))
    #results = pkl.load(open('opt/class_3_3_perbenzacid_neural1_50eps_results.dat'))
    cmat = results['confusion_matrix'][1]

    #results = pkl.load(open('baseline/pba_bl_rgt_rct2_cf.dat'))
    #cmat = results
    print type(cmat)
    short_labels = ['NR', 'SN', 'Elim.', 'SN+m', 'E+m', 'H-X M.', 'H-X AntiM.', 
        'H2O M.', 'H2O AntiM.', 'R-OH (M.)', 'H2', 'X-X Add.', 'X-OH Add.', 'Epox.', 
        'Hydrox.', 'Ozon.', 'Polymer.']
    full_labels = ['Null Reaction', 'Nucleophilic substitution', 'Elimination', 
        'Nucleophilic Substitution with Methyl Shift', 'Elimination with methyl shift', 
        'Hydrohalogenation (Markovnikov)', 'Hydrohalogenation (Anti-Markovnikov)', 
        'Hydration (Markovnikov)', 'Hydration (Anti-Markovnikov)', 'Alkoxymercuration-demercuration', 
        'Hydrogenation', 'Halogenation', 'Halohydrin formation', 'Epoxidation', 
        'Hydroxylation', 'Ozonolysis', 'Polymerization']
    num_labels = [str(i) for i in range(len(full_labels))]
    labelled_cmat(cmat, num_labels, rotation=0) 

