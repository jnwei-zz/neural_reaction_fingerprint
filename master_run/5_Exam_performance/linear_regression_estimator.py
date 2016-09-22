'''
Updated reaction estimator, but for fp_method_0
    i.e. adding all the fingerprints together

2/12 : Now also updated for neural fingerprints

#!# Check: current settings, set to allow fingerprint to vary for testing 
'''


import autograd.numpy as np
from autograd.scipy.misc import logsumexp
import autograd.numpy.random as npr
from autograd import grad
from sklearn.base import BaseEstimator, RegressorMixin
from rdkit import Chem
from rdkit.Chem import AllChem
import inspect
from sklearn.metrics import confusion_matrix as conf_mat

#from parse_data import split_smiles, get_train_test_sets
from neuralfingerprint import build_standard_net, relu
from neuralfingerprint import build_triple_conv_deep_net, adam
from neuralfingerprint import build_batched_grad, categorical_nll 

class linreg_estimator(BaseEstimator, RegressorMixin):
    '''
    Makes predictions about reactions using classification 
    '''
    
    def __init__(self,
                    log_learn_rate = -5., 
                    log_init_scale = -6.,
                    fp_length = 10,
                    other_param_dict = {'num_epochs' : 100,
                    'batch_size' : 200, 'normalize'  : 1,
                    'dropout'    : 0, 'activation' :relu,
                    'num_outputs': 18,  'init_bias': 0.85}): 
                    #log_l1_penalty= 10e-5,  log_l2_penalty= 10e-5):
        
        args, _, _, values = inspect.getargvalues(inspect.currentframe())

        for arg, val in values.items(): 
            setattr(self, arg, val)


    def parse_training_params(self):
        nn_train_params = {'num_epochs'  : self.other_param_dict['num_epochs'],
                           'batch_size'  : self.other_param_dict['batch_size'],
                           'learn_rate'  : np.exp(self.log_learn_rate),
                           'param_scale' : np.exp(self.log_init_scale)}
    
        vanilla_net_params = {'layer_sizes': [self.fp_length],
                              'normalize':  self.other_param_dict['normalize'],
                              #'L2_reg': np.exp(self.log_l2_penalty),
                              #'L1_reg': np.exp(self.log_l1_penalty),
                              'activation_function': self.other_param_dict['activation'],
                              'nll_func' : categorical_nll,
                              'num_outputs' : self.other_param_dict['num_outputs']}
        return nn_train_params, vanilla_net_params


    def train_nn(self, pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, train_params,
                 validation_smiles=None, validation_raw_targets=None):
    #def train_nn(self, pred_fun, loss_fun, num_weights, train_fps, train_raw_targets, train_params,
    #             validation_smiles=None, validation_raw_targets=None):
        """loss_fun has inputs (weights, smiles, targets)"""
        print "Total number of weights in the network:", num_weights
        npr.seed(0)
        init_weights = npr.randn(num_weights) * train_params['param_scale']
        init_weights[-1] = self.other_param_dict['init_bias']
    
        #train_targets, undo_norm = normalize_array(train_raw_targets)
        training_curve = []
        def callback(weights, iter):
            if iter % 20 == 0:
                print "max of weights", np.max(np.abs(weights))
                #train_preds = undo_norm(pred_fun(weights, train_smiles))
                cur_loss = loss_fun(weights, train_smiles, train_raw_targets)
                #cur_loss = loss_fun(weights, train_fps, train_raw_targets)
                training_curve.append(cur_loss)
                print "Iteration", iter, "loss", cur_loss, 
    
        grad_fun = grad(loss_fun)
        #grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
        #                                        train_fps, train_raw_targets)
        grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                                train_smiles, train_raw_targets)
    
        #num_iters = train_params['num_epochs'] * np.shape(train_fps)[0] / train_params['batch_size']
        num_iters = train_params['num_epochs'] * np.shape(train_smiles)[0] / train_params['batch_size']
        trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                               num_iters=num_iters, step_size=train_params['learn_rate'])
                               #b1=train_params['b1'], b2=train_params['b2'])
    
        def predict_func(new_smiles):
            """Returns to the original units that the raw targets were in."""
            return pred_fun(trained_weights, new_smiles)
        
        def predict_func_fps(new_fps):
            """ return function for fps """
            return pred_fun(trained_weights, new_fps)

        return predict_func, trained_weights, training_curve


    def fit(self, X, y):
        '''
        #X - list of smiles
        X - fp array 
        y - numerical value to calculate
        '''
        #start = time.time()
        nn_train_params, vanilla_net_params = self.parse_training_params()
        print "Task params", nn_train_params, vanilla_net_params

        print "Building custom fingerprint wtih rct and rgt"
        loss_fun, pred_fun, net_parser = build_standard_net(**vanilla_net_params)

        # rewrite so using pre-built fps 
        num_weights = len(net_parser)
        predict_func, trained_weights, conv_training_curve = \
            self.train_nn(pred_fun, loss_fun, num_weights, X, y,  
                     nn_train_params)
        
        self.predict_func = predict_func
        self.train_weights = trained_weights
        self.net_parser = net_parser

    def getWeights(self):
        #@pre: fit has been run once.
        return self.train_weights, self_parser


    def predict(self, X, y=None):
        # @pre: training weights have to be set with fit 
        return self.predict_func(X)


    def score(self, X, y):
        err = categorical_nll( self.predict(X), y,  self.other_param_dict['num_outputs']) 
        return err 

    def accuracy(self, X, y):
        preds = self.predict(X)
        isMaxPred = [[val == max(row) for val in row] for row in preds] 
        isMaxTarg = [[val == max(row) for val in row] for row in y] 
        return float(sum([isMaxPred[ii] == isMaxTarg[ii] for ii in range(len(preds))]))/len(preds)

