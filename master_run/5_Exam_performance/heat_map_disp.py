'''

This code is to be used for displaying heat maps of prediction results
  i.e. show the probability of the correct reaction type as predicted by each algorithm 
'''

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle as pkl

#matplotlib.use('tkagg')


#column_labels = ['Reagent baseline', 'Reactant2 baseline', 'Reagent+Reactant2 baseline', 'Morgan fingerprints', 'neural fingerprint'] 
#column_labels = ['Baseline', 'Morgan Fingerprint', 'Neural Fingerprint'] 
column_labels = ['Baseline', 'Morgan', 'Neural'] 


##Problem 8-47
#data = pkl.load(open('class_3_3_nopba_Wade8_47_50opt.dat'))[::-1]
#row_labels = list('abcdefghijlmnop')
#row_labels = ['8-47'+i for i in row_labels]

data = pkl.load(open('class_3_3_nopba_Wade8_48_50opt.dat'))[::-1]
# Add header:
row_labels = list('abcdefg')
row_labels = ['8-48'+i for i in row_labels]

fig, ax = plt.subplots()
sns.heatmap(data, ax=ax, annot=True, fmt=".3f", cmap='YlGn', vmin = 0, vmax = 1)

ax.xaxis.tick_top()
ax.set_xticklabels(column_labels, minor=False,  wrap=True, ha='center')
ax.set_yticklabels(row_labels, minor=False, rotation=0)
ax.invert_yaxis()

#plt.savefig('class_3_3_nopba_Wade8_47_50opt.svg')
plt.savefig('class_3_3_nopba_Wade8_48_50opt.svg')
#plt.show()
