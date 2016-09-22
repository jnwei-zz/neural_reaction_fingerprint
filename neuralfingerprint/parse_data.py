'''
Library for parsing data:
'''
import csv
import numpy as np
from scipy.sparse import csr_matrix
from scipy.misc import logsumexp


def read_csv_all(filename, input_name, target_name_arr=[]):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        #for row in it.islice(reader, nrows):
        for row in reader:
            data[0].append(row[input_name])
            #data[1].append(float(row[target_name]))
            # Try to get all the vectors 
            targets_list = []
            for kwarg in target_name_arr:
                targets_list.append(float(row[kwarg]))
            data[1].append(targets_list)
    return map(np.array, data)

def load_data_noslice(filename, input_name, target_name_arr):
    data = read_csv_all(filename, input_name, target_name_arr)
    return (data[0], data[1])

def slicetuple(d, ixs):
    return tuple([v[ixs] for v in d])

def get_train_test_sets(dat_file, input_name, target_name_arr, N_train):
    np.random.seed(42)
    print "Loading data..."
    all_rxn  = load_data_noslice(dat_file, input_name, target_name_arr)

    # Permute the data strings
    N_data = len(all_rxn[0])
    random_main_idxs = np.random.permutation(N_data)
    train_idxs = random_main_idxs[:N_train]
    test_idxs = random_main_idxs[N_train:]

    return slicetuple(all_rxn, train_idxs), slicetuple(all_rxn, test_idxs)

def split_smiles(input_smiles):
    smile1_list = ['' for i in range(len(input_smiles))]
    smile2_list = ['' for i in range(len(input_smiles))]

    for i in range(len(input_smiles)):
        try:
            smi1, smi2 = input_smiles[i].split('.')
            #print smi1
            smile1_list[i] = smi1
            smile2_list[i] = smi2
        except: print 'error no dot found in reaction string'

    return zip(smile1_list, smile2_list)

def split_smiles_triples(input_smiles):
    smile1_list = ['' for i in range(len(input_smiles))]
    smile2_list = ['' for i in range(len(input_smiles))]
    smile3_list = ['' for i in range(len(input_smiles))]
    smile4_list = ['' for i in range(len(input_smiles))]
    
    for i in range(len(input_smiles)):
        try:
            rcts_smi, rgt_smi, prod_smi = input_smiles[i].split('>')
            smi1, smi2 = rcts_smi.split('.')
            #print smi1
            smile1_list[i] = smi1
            smile2_list[i] = smi2

            if rgt_smi == 'hv': rgt_smi = rgt_smi.replace('hv', '[H][V]')
            smile3_list[i] = rgt_smi
            smile4_list[i] = prod_smi

        except: 
            print 'error no dot found in reaction string'
            print input_smiles[i]
            break
    # Return all possible outputs in order:
    #       rcts + rgts,  rcts (all) + prods, rcts (1) + prods (only use for no swap)
    return zip(smile1_list, smile2_list, smile3_list), zip(smile1_list, smile2_list, smile4_list), zip(smile1_list, smile4_list)


def strip_smiles_carrots(input_smiles):
    # For stripping smiles of carrots and 'hv' 

    new_smi_list = [''] * len(input_smiles)

    for i in range(len(input_smiles)):
        new_smi_list[i] = str(input_smiles[i]).replace('>','.')

        if 'hv' in new_smi_list[i]:
            new_smi_list[i] = new_smi_list[i].replace('hv', '[H][V]')

        if '..' in new_smi_list[i]:
            new_smi_list[i] = new_smi_list[i].replace('..', '.[Nd].')
    
        new_smi_list[i] = str(new_smi_list[i]).rstrip('.')

    return new_smi_list


def confusion_matrix(true_mat, unnorm_pred_mat, num_outputs):
    # Calculating confusion matrix based on accuracy of predictions 
    # returns an num_outputs x num_outputs matrix

    # true reaction type on the horizontal axis
    # predicted reaction on the vertical axis

    csr_true_mat = csr_matrix(true_mat)
    conf_mat = np.zeros((num_outputs,num_outputs))
    row_idxs, col_idxs =  csr_true_mat.nonzero()
    #print row_idxs
    #print col_idxs

    # Note: Raw predictions made in unnormalized form, adjust this
    pred_mat = np.exp(unnorm_pred_mat  - logsumexp(unnorm_pred_mat, axis=1, keepdims=True))
    rxn_ct = np.ones(num_outputs)*0.0001

    for rxn_subidx in range(len(row_idxs)):
        conf_mat[:,col_idxs[rxn_subidx]] += csr_true_mat[row_idxs[rxn_subidx],col_idxs[rxn_subidx]]*pred_mat[row_idxs[rxn_subidx],:]
        rxn_ct[col_idxs[rxn_subidx]] += 1.*csr_true_mat[row_idxs[rxn_subidx],col_idxs[rxn_subidx]]

    conf_mat = conf_mat/rxn_ct

    print 'Test normalization of columns: '
    print sum(conf_mat) 

    # Also return number of each type of reaction sampled
    return rxn_ct, conf_mat


def get_normalized_pred(unnorm_pred_mat):
    # Normalizes the prediction matrix (which is currently reported as log of probability and unormalized
    pred_mat = np.exp(unnorm_pred_mat  - logsumexp(unnorm_pred_mat, axis=1, keepdims=True))
    return pred_mat

def accuracy(preds, targets):
    # Calculate the accuracy of the prediction
    # based on matches in max values in the true and predicted probability vectors 
    isMaxPred = [[val == max(row) for val in row] for row in preds]
    isMaxTarg = [[val == max(row) for val in row] for row in targets]
    return float(sum([isMaxPred[ii] == isMaxTarg[ii] for ii in range(len(preds))]))/len(preds)


def L1_error(preds, targets, num_outputs):
    # NOTE: preds matrix gives log likelihood of result, not likelihood probability
    #      raise to exponential to get correct value
    # Use Brier score for error estimate

    preds = preds - logsumexp(preds, axis=1, keepdims=True)
    pred_probs = np.exp(preds)

    return np.mean(np.linalg.norm(pred_probs - targets, axis=1))
