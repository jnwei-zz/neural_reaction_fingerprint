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
from scipy.misc import logsumexp


num_outputs = 18
other_param_dict = {'num_epochs' : 10,
                    'batch_size' : 100, 'normalize'  :0 ,
                    'dropout'    : 0, 'fp_depth': 4, 'activation' :relu, 
                    'fp_type' : 'neural',
                    #'fp_type' : 'morgan', 
                    'h1_size' : 100, 'conv_width': 20, 'num_outputs': num_outputs, 
                    'init_bias': 0.85} 

#learn rate, init scale,
# Morgan 1: 
# Change to correct value (I deleted unfortunately...)
#RE = rxn_estimator(-2.9980333058185202, -4.92986261217202, 153, other_param_dict)

# Morgan 2:
#RE = rxn_estimator(-3.7771145904698447, -4.3688416048512435, 752, other_param_dict)

# Morgan 3:
RE = rxn_estimator(-3.9815288594632183,-3.5896897128869263 ,767 , other_param_dict)

# Neural 1:
#RE = rxn_estimator( -3.2721919214871984, -2.3649628009746966, 148, other_param_dict) 

#Neural 2:
#RE = rxn_estimator(-3.6682766762789663, -3.1635758381500736, 156, other_param_dict)


train_input_1 = pkl.load(open("/home/jennifer/Documents/DeepMolecules-master/reaction_learn/Classification_3_1/data/200each/balanced_200each_train_inputs_1.dat"))
train_targets = pkl.load(open("/home/jennifer/Documents/DeepMolecules-master/reaction_learn/Classification_3_1/data/200each/balanced_200each_train_targets.dat"))

test_input_1 = pkl.load(open("/home/jennifer/Documents/DeepMolecules-master/reaction_learn/Classification_3_1/data/200each/balanced_200each_test_inputs_1.dat"))
test_targets = pkl.load(open("/home/jennifer/Documents/DeepMolecules-master/reaction_learn/Classification_3_1/data/200each/balanced_200each_test_targets.dat"))


print 'beginning training'
start = time.time()

RE.fit(train_input_1, train_targets)

print 'end of training'
print 'run time: ', time.time() - start

test_predictions = RE.predict(test_input_1)

def cat_error(preds, targets, num_outputs):
    # NOTE: preds matri gives log likelihood of result, not likelihood probability
    #      raise to exponential to get correct value
    # Use Brier score for error estimate

    preds = preds - logsumexp(preds, axis=1, keepdims=True)
    pred_probs = np.exp(preds)

    return np.mean(np.linalg.norm(pred_probs - targets, axis=1))

def accuracy(preds, targs):
    isMaxPred = [[val == max(row) for val in row] for row in preds] 
    isMaxTarg = [[val == max(row) for val in row] for row in targs] 
    return float(sum([isMaxPred[ii] == isMaxTarg[ii] for ii in range(len(preds))]))/len(preds)

#print test_predictions[:10]
# save prediction:

#with open('morgan1_balanced_200each_preds.dat','w') as f:
with open('neural1_balanced_200each_preds.dat','w') as f:
    pkl.dump(test_predictions, f)

print 'train_NLL: ', RE.score(train_input_1, train_targets)
print 'train accuracy: ', accuracy(RE.predict(train_input_1), train_targets)

score= RE.score(test_input_1, test_targets)
conf_matrix = conf_mat(test_targets, test_predictions,num_outputs)
l1_error = cat_error(test_predictions, test_targets, num_outputs)
accuracy = accuracy(test_predictions, test_targets)

results = {'cross_entropy_score':score,'confusion_matrix':conf_matrix, 'l1_norm':l1_error, 'accuracy':accuracy}

#with open('morgan1_balanced_200each_results.dat','w') as f:
with open('neural1_balanced_200each_results.dat','w') as f:
    pkl.dump(results, f)
