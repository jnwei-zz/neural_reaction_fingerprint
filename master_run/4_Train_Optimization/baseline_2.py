'''
1. Count number of each rxn type in train set; use this as baseline

2. Find 20 most common reagents (just reagents)
    - Map these to their most common rxn num 
    - Build a dictionary from this and predict outcomes
 
# seems to work for now, but some discrepancies in the dataset

'''
from collections import Counter
#from nn_baseline_estimator import neuralnet_estimator as NN_estimator 
from linear_regression_estimator import linreg_estimator as LR_estimator 
from neuralfingerprint import relu, categorical_nll
from neuralfingerprint.parse_data import confusion_matrix
import pickle as pkl
import numpy as np

def rct_rgt_lists( num_rgts_observe, num_rct2s_observe, train_rxns_parsed):
    # @brief: Make a list of most common rct2s and rgts from datasets.
    # Parameters:
    # num_rgts-observe, num_rct2s_observe  - number of rgts/rcts to observe
    # input_smis: input smiles and targets; read from data files
    rgtCounter = Counter()
    rct2Counter = Counter()
    
    for rxn in train_rxns_parsed:
        _, rcts2, rgts = rxn
        rxn_rgt_list = rgts.split('.')
        rxn_rct2_list = rcts2.split('.')
        for rgt in rxn_rgt_list:
            rgtCounter[rgt] += 1
        for rct2 in rxn_rct2_list:
            rct2Counter[rct2] += 1
    
    rgt_list = [rgt_smi for rgt_smi, rgt_frq in rgtCounter.most_common(num_rgts_observe+1)[1:]] # omit default 'COC', which would be most common rgt
    rct2_list = [rct2_smi for rct2_smi, rct2_frq in rct2Counter.most_common(num_rct2s_observe)]

    #Need to store these to make fingerprints
    return rgt_list, rct2_list
   

def make_rct_rgt_fps(rgt_list, rct2_list, input_smis):
    # Make new fingerprint based on one-hot for reagents
    # fp_length = num_rgts_observe + 1
    train_rgt_fp = np.zeros((len(input_smis), num_rgts_observe+1))
    train_rct2_fp = np.zeros((len(input_smis), num_rct2s_observe+1))
    
    # Will need one also for reactants as well
    
    for ii in range(len(train_targs)):
        # parse the smiles, see if find one of entries from list
        # if it is in the list, great, add to list.
        # if not, add to last to last slot
    
        _, rcts2, rgts = input_smis[ii]
        rxn_rgt_list = rgts.split('.')
        rxn_rct2_list = rcts2.split('.')
        
        # Turn on fingerprint for those
        rgt_found_flag = False
        for rgt_idx in range(len(rgt_list)):
            if rgt_list[rgt_idx] in rxn_rgt_list:
                train_rgt_fp[ii][rgt_idx] = 1.0
                rgt_found_flag = True 
        if rgt_found_flag == False:
            train_rgt_fp[ii][-1] = 1.0
        
        rct2_found_flag = False
        for rct2_idx in range(len(rct2_list)):
            if rct2_list[rct2_idx] in rxn_rct2_list:
                train_rct2_fp[ii][rct2_idx] = 1.0
                rct2_found_flag = True 
        if rct2_found_flag == False:
            train_rct2_fp[ii][-1] = 1.0
   
    # Return fingerprints associated with reaction fingerprints.
    return train_rgt_fp, train_rct2_fp

# With the fingerprints made, begin linear regression training
#Make use of the training code

if __name__ == '__main__':

    num_rgts_observe = 20
    num_rct2s_observe = 30

    with open('../../data/200each/balanced_200each_train_inputs_1.dat') as train_set_f:
        train_rxns_parsed = pkl.load(train_set_f) 
    with open('../../data/200each/balanced_200each_train_targets.dat') as train_set_targf:
        train_targs = pkl.load(train_set_targf) 
    
    with open("../../data/200each/balanced_200each_test_inputs_1.dat") as test_set_f:
        test_rxns_parsed = pkl.load(test_set_f)
    with open("../../data/200each/balanced_200each_test_targets.dat") as test_set_targf:
        test_targs = pkl.load(test_set_targf)

    other_param_dict = {'num_epochs' : 50,
                    # Production run? Keep batch size same for now
                    'batch_size' : 100, 'normalize'  :1 ,
                    'dropout'    : 0, 'fp_depth': 4, 'activation' :relu, 
                    'num_outputs': 17, 'h1_size': 100, 
                    'init_bias': 0.85} 
   
    # Getting most common rgts and rct2s
    rgt_list, rct2_list = rct_rgt_lists(num_rgts_observe, num_rct2s_observe, train_rxns_parsed)
    with open('rgt_baseline_list.dat','w') as f:
        for rgt in rgt_list:
            f.write(rgt+'\n')
    
    with open('rct2_baseline_list.dat','w') as f:
        for rct2 in rct2_list:
            f.write(rct2+'\n')

    # Making fps from the lists
    train_rgt_fp, train_rct2_fp = make_rct_rgt_fps( rgt_list, rct2_list, train_rxns_parsed)
    test_rgt_fp, test_rct2_fp = make_rct_rgt_fps( rgt_list, rct2_list, test_rxns_parsed)

    ## Learn rate, init scale, fp length
    
    #clf = NN_estimator( -4, -4, num_rgts_observe+num_rct2s_observe, other_param_dict)
    #clf = NN_estimator( -4, -4, num_rgts_observe, other_param_dict)
    #clf = NN_estimator( -4, -4, num_rct2s_observe, other_param_dict)

    def baseline_train_func(train_inps, train_targs, test_inps, test_targs, log_learn_rate, log_init_scale, fp_length):
        # Return confusion matrix and trainign weights
        clf = LR_estimator( log_learn_rate, log_init_scale, fp_length, other_param_dict)
        clf.fit(train_inps, train_targs)

        # Results
        # Training NLL score:
        print '\nNLL score: ', clf.score(train_inps, train_targs)
        print 'Train accuracy: ', clf.accuracy(train_inps, train_targs)

        # test the prediction
        print 'Test NLL score: ', clf.score(test_inps, test_targs) 
        print 'Test accuracy: ', clf.accuracy(test_inps, test_targs) 

        # Calculate confusion matrix
        conf_mat = confusion_matrix(test_targs, clf.predict(test_inps), other_param_dict['num_outputs'])
        return conf_mat[1]

    train_rgt_rct2_fp = np.concatenate((train_rgt_fp, train_rct2_fp), axis=1)
    test_rgt_rct2_fp = np.concatenate((test_rgt_fp, test_rct2_fp), axis=1)

    # Calculating confusion matrix for results
    cf_mat = baseline_train_func(train_rgt_rct2_fp, train_targs, test_rgt_rct2_fp, test_targs, -4, -4, num_rct2s_observe+num_rgts_observe+2)
    with open('rgt_rct2_cf.dat','w') as f:
        pkl.dump(cf_mat, f)
