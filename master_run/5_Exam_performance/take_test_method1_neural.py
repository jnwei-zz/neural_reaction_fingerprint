#Testing a single run:

# load dataset
# set best parameters
# run


from fp1_reaction_estimator import rxn_estimator
from neuralfingerprint import relu
import pickle as pkl
import time
import numpy as np
from neuralfingerprint.parse_data import confusion_matrix as conf_mat
from neuralfingerprint.parse_data import split_smiles_triples
from scipy.misc import logsumexp

retrain = True

num_outputs = 18
other_param_dict = {'num_epochs' : 50,
                    'batch_size' : 200, 'normalize'  :1 ,
                    'dropout'    : 0, 'fp_depth': 4, 'activation' :relu, 
                    #'fp_type' : 'morgan',
                    'fp_type' : 'neural',
                    'save_weights' : False,
                    'h1_size' : 100, 'conv_width': 20, 'num_outputs': num_outputs, 
                    'init_bias': 0.85} 

#learn rate, init scale,

# Neural 1, 100 runs, norm
# Parameters are : learn rate, init scale, and fingerpritn length
RE = rxn_estimator( -3.7708613280344383, -3.5680734313359697, 142, other_param_dict) 

train_input_1 = pkl.load(open("../../data/200each/balanced_200each_train_inputs_1.dat"))
train_targets = pkl.load(open("../../data/200each/balanced_200each_train_targets.dat"))


print 'beginning training'
start = time.time()

RE.fit(train_input_1, train_targets)

print 'end of training'
print 'run time: ', time.time() - start

batch_nm = 'exam_200each'
method = other_param_dict['fp_type']+'1' 
problem_list = ['Wade8_47', 'Wade8_48']

for prob in problem_list:
    with open('../../data/test_questions/'+ prob+'.cf.txt') as f1:
        input_smiles = f1.readlines()

    test_input_1 , _, _ = split_smiles_triples(input_smiles)   
    test_predictions = RE.predict(test_input_1)
        
    with open('results/'+batch_nm+'_'+method+'_'+prob+'.dat','w') as resf:
        pkl.dump(test_predictions, resf)

