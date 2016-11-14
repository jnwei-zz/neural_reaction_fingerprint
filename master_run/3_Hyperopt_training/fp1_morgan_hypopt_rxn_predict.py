'''
Same set up, but for a neural fingerprint.

All reactants are combined into a neural fingerprint

# Checklist:
1. Set fingerprint method
2. Set output file
3. Set task_params and batch size
4. Set number of runs

'''

from fp1_reaction_estimator import rxn_estimator
import autograd.numpy as np
from sklearn.cross_validation import cross_val_score
from neuralfingerprint.parse_data import get_train_test_sets, split_smiles_triples
from neuralfingerprint import relu, smiles_to_fps
import pickle as pkl
import time

from hyperopt import fmin, hp, tpe, STATUS_OK, Trials


target_name_array = ['prob_'+str(i) for i in range(17)]
print target_name_array


task_params  = {'smiles_name' : 'smiles', 
               'target_name_arr' : target_name_array} 


other_param_dict = {'num_epochs' : 20,
                    # Production run? Keep batch size same for now
                    'batch_size' : 100, 'normalize'  :1,
                    'dropout'    : 0, 'fp_depth': 4, 'activation' :relu, 
                    #'fp_type' : 'neural',
                    'fp_type' : 'morgan', 
                    'h1_size' : 100, 'conv_width': 20, 'num_outputs': 17, 
                    'init_bias': 0.85} 

# Production run:
max_num_runs  = 100 
        
X = pkl.load(open('train_inputs.dat')) 
y = pkl.load(open('train_targets.dat')) 

task_params['N_train'] = np.shape(X)[0]

## Set cross validation parameters here.
def hyperopt_train_test(params):
    clf = rxn_estimator(np.float32(params[0]), np.float32(params[1]), np.int(params[2]), other_param_dict)
    return cross_val_score(clf, X, y, cv=3).mean()

run_counter = 0
def myFunc(params):
    print '########################'
    global run_counter
    print '{} run out of {}'.format(run_counter+1, max_num_runs)

    start_time = time.time()
    print params
    acc= hyperopt_train_test(params)

    print '\nend time: {}'.format(time.time() - start_time)
    run_counter += 1
    return {'loss': acc, 'status':STATUS_OK }


#!# uniform fp length? Should be integer...
train_space = (hp.uniform('log_learn_rate', -4, -1.5),
               hp.uniform('log_init_scale', -5, -2),
               # for morgan: 
               hp.quniform('fp_length', 10, 1024 ,1))
               # for neural: 
               #hp.quniform('fp_length', 200))

if __name__ == '__main__':
#if False:
    print task_params 
    print other_param_dict
    #print train_fps
    trials = Trials()
    best = fmin( fn = myFunc, space = train_space, algo=tpe.suggest, max_evals = max_num_runs, trials = trials)
    
    print 'best:', best
    print 'trials:'
    for trial in trials.trials[:2]:
        print trial


    #!# Set every time
    with open('morgan1_bal_200each_100run_norm_train.dat','w') as resultf:
    #with open('morgan2_bal_200each_100run_norm_train.dat','w') as resultf:
        pkl.dump(trials, resultf)

