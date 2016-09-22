#Testing a single run:

# load dataset
# set best parameters
# run


#from fp1_reaction_estimator import rxn_estimator
# Set for baseline 2 with neural net
#from nn_baseline_estimator import neuralnet_estimator as NN_estimator 
from linear_regression_estimator import linreg_estimator as LR_estimator
import pickle as pkl
import time
import numpy as np
from neuralfingerprint import relu
from neuralfingerprint.parse_data import confusion_matrix as conf_mat
from neuralfingerprint.parse_data import split_smiles_triples
from baseline_1 import make_rct_rgt_fps 
from scipy.misc import logsumexp

num_outputs = 18
other_param_dict = {'num_epochs' : 50,
                    'batch_size' : 100, 'normalize'  :1 ,
                    'h1_size' : 100,
                    'num_outputs': num_outputs, 'activation':relu, 
                    'init_bias': 0.85} 

# learning training data
train_input_1 = pkl.load(open("../../data/200each/balanced_200each_train_inputs_1.dat"))
train_targets = pkl.load(open("../../data/200each/balanced_200each_train_targets.dat"))

# Must run baseline_1.py in ../4_Train_Optimization/ to get these files
# These are the lists of the most common reagent and reactants
with open('../4_Train_Optimization/rgt_baseline_list.dat') as f:
    temp_rgt_list = f.readlines()
    rgt_list = [rgt.strip() for rgt in  temp_rgt_list]
        
with open('../4_Train_Optimization/rct2_baseline_list.dat') as f:
    temp_rct2_list = f.readlines()
    rct2_list = [rct2.strip() for rct2 in  temp_rct2_list]


def make_rxn_fps(bline_rgt_fp, bline_rct_fp, bl_type):
    # Assume the default, i.e. examine 30 most common reagents and 20 most common secondary reactants
    baseline_fp = np.concatenate((bline_rgt_fp, bline_rct_fp), axis=1)
    fp_len = 52
    return baseline_fp, fp_len 
        

batch_nm = 'exam_200each'

problem_list = ['Wade8_47', 'Wade8_48'] 
method_dict = {'bl_rgt_rct2':3}


bline_rgt_fp, bline_rct_fp = make_rct_rgt_fps(rgt_list, rct2_list, train_input_1)  

for method in method_dict.keys():
    print 'beginning training'
    start = time.time()
    train_fp, fp_length = make_rxn_fps(bline_rgt_fp, bline_rct_fp, method_dict[method])
    LR = LR_estimator(-4, -4 , fp_length, other_param_dict)
    LR.fit(train_fp, train_targets)

    print 'end of training'
    print 'run time: ', time.time() - start
    
    for prob in problem_list:
        with open('../../data/test_questions/'+ prob+'.cf.txt') as f1:
            input_smiles = f1.readlines()
            test_input_1 , _, _ = split_smiles_triples(input_smiles)   
    
        bline_test_rgt_fp, bline_test_rct_fp = make_rct_rgt_fps(rgt_list, rct2_list, test_input_1)  
        test_input, fp_length = make_rxn_fps(bline_test_rgt_fp, bline_test_rct_fp, method_dict[method])
        test_predictions = LR.predict(test_input)

        with open('../results/'+batch_nm+'_'+method+'_'+prob+'.dat','w') as resf:
            pkl.dump(test_predictions, resf)


