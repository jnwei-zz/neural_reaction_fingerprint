import pickle as pkl
import sys

fname  = sys.argv[1]
A = pkl.load(open(fname))
min_idx = A.losses().index(min(A.losses()))
print A.trials[min_idx]


