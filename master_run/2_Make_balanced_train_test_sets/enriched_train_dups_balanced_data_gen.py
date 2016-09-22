'''
In this version, generating datasets that have same number of training and test sets
Achieves this by duplicating reactions in the training and test sets.

Main parameters to set: 
num_per_rxn  -- number of train reactions for each rxn type
    - Remember to change name of dataset folders 
num_test_rxn -- number of test reactions for each rxn type

@pre should already have all reactions in balanced_set dir, each file is a reaction type. 

'''
from os import listdir
from os.path import isfile, join
import autograd.numpy as np
from neuralfingerprint.parse_data import get_train_test_sets, strip_smiles_carrots, split_smiles_triples, load_data_noslice, load_data_noslice 
import pickle as pkl

dat_file_names = [f for f in listdir('./balanced_set') if isfile(join('./balanced_set',f))]
#dat_file_names = ['alkylhal_NR_rxn.dat', 'E_rxn.dat', 'Em_rxn.dat', 'polymer.dat']

num_outputs = 17
target_name_array = ['prob_'+str(i) for i in range(num_outputs)]

num_per_rxn = 200
num_test_rxn = 1000
train_inputs = []
train_targets = []
test_inputs = []
test_targets = []
np.random.seed(42)

def space_fill(data_set_tup, fill_amt):
    # @inp data_set_tup - tuple with two entries, [0] is smiles, [1] is target
    # @inp fill_amt - amount of space to fill up with reactions
    # @return : two lists of smis and target arrays

    sub_num_rxn = len(data_set_tup[1])
    num_copies = fill_amt/ sub_num_rxn 
    remainder = fill_amt % sub_num_rxn 

    # Randomly select remainder:
    sfill_ids = [idx for idx in range(sub_num_rxn)]
    np.random.shuffle(sfill_ids)

    rand_remain_ids = sfill_ids[:remainder]

    # concatenate together:
    tup_input = tuple([data_set_tup[0] for i in range(num_copies)])
    tup_input += (data_set_tup[0][rand_remain_ids],)

    tup_target = tuple([data_set_tup[1] for i in range(num_copies)])
    tup_target += (data_set_tup[1][rand_remain_ids],)

    return np.concatenate(tuple(tup_input), axis=0), np.concatenate(tuple(tup_target), axis=0) 

for dat_file_nm in dat_file_names: 

    inp_smi, targs = load_data_noslice('./balanced_set/'+dat_file_nm, input_name='smiles', 
                target_name_arr=target_name_array)

    rxn_ct = len(inp_smi)

    # If the number of reactions in the set is more than that required
    if rxn_ct > num_per_rxn + num_test_rxn :
        # already preshuffled
        df_train_set, df_test_set = get_train_test_sets('./balanced_set/'+dat_file_nm, input_name='smiles',
            target_name_arr=target_name_array, N_train=num_per_rxn)
    
        train_inputs.append(df_train_set[0])
        train_targets.append(df_train_set[1])

        # Append a part of the remaining test set to the data
        test_inputs.append(df_test_set[0][:num_test_rxn])
        test_targets.append(df_test_set[1][:num_test_rxn])

    # If the number of reactions in the set is fewer than the number of train rxns and test rxn req. 
    elif rxn_ct < 1.2*num_per_rxn: 
        print dat_file_nm + ' has fewer than ' + str(num_per_rxn)
       
        # split these reactions in half:
        df_train_set, df_test_set = get_train_test_sets('./balanced_set/'+dat_file_nm, input_name='smiles',
            target_name_arr=target_name_array, N_train=rxn_ct/2)

        # Create new data sets that are space-filled versions to match required length
        new_dup_train_inputs, new_dup_train_targs = space_fill(df_train_set, num_per_rxn)
        train_inputs.append(new_dup_train_inputs)
        train_targets.append(new_dup_train_targs)

        new_dup_test_inputs, new_dup_test_targs = space_fill(df_test_set, num_test_rxn )
        test_inputs.append(new_dup_test_inputs)
        test_targets.append(new_dup_test_targs)

    else:
        print dat_file_nm + ' has fewer than ' + str(num_per_rxn + num_test_rxn) 

        df_train_set, df_test_set = get_train_test_sets('./balanced_set/'+dat_file_nm, input_name='smiles',
            target_name_arr=target_name_array, N_train=num_per_rxn)

        train_inputs.append(df_train_set[0])
        train_targets.append(df_train_set[1])

        new_dup_test_inputs, new_dup_test_targs = space_fill(df_test_set, num_test_rxn )
        test_inputs.append(new_dup_test_inputs)
        test_targets.append(new_dup_test_targs)


# Combine the datasets:
# Need to concatenate, not just add tuples

train_set_200each_inputs = np.concatenate(tuple(train_inputs), axis=0) 
train_set_200each_targets = np.concatenate(tuple(train_targets), axis=0) 
test_set_200each_inputs = np.concatenate(tuple(test_inputs), axis=0) 
test_set_200each_targets = np.concatenate(tuple(test_targets), axis=0) 

train_idxs = [idx for idx in range(len(train_set_200each_inputs))]
test_idxs =  [idx for idx in range(len(test_set_200each_inputs))]

np.random.shuffle(train_idxs) 
np.random.shuffle(test_idxs) 

train_set_200each_inputs = train_set_200each_inputs[train_idxs] 
train_set_200each_targets = train_set_200each_targets[train_idxs]
test_set_200each_inputs = test_set_200each_inputs[test_idxs] 
test_set_200each_targets = test_set_200each_targets[test_idxs]

train_set_200each_inputs_0 = strip_smiles_carrots(train_set_200each_inputs)

# 1: rct1 + rct2 + rgts
# 2: rct1 + rct2 + prod
# 3: rct1 + prod
train_set_200each_inputs_1, train_set_200each_inputs_2, train_set_200each_inputs_3 = split_smiles_triples(train_set_200each_inputs)

test_set_200each_inputs_0 = strip_smiles_carrots(test_set_200each_inputs)
test_set_200each_inputs_1, test_set_200each_inputs_2 , test_set_200each_inputs_3 = split_smiles_triples(test_set_200each_inputs)
# Look at method 2

# Save the data sets:

with open('200each/balanced_200each_train_inputs_0.dat','w') as f:
#with open('test_balanced_200each_train_inputs_0.dat','w') as f:
    pkl.dump(train_set_200each_inputs_0,f)

with open('200each/balanced_200each_train_inputs_1.dat','w') as f:
    pkl.dump(train_set_200each_inputs_1,f)

with open('200each/balanced_200each_train_inputs_2.dat','w') as f:
    pkl.dump(train_set_200each_inputs_2,f)

with open('200each/balanced_200each_train_inputs_3.dat','w') as f:
    pkl.dump(train_set_200each_inputs_3,f)

with open('200each/balanced_200each_train_targets.dat','w') as f:
#with open('test_balanced_200each_train_targets.dat','w') as f:
    pkl.dump(train_set_200each_targets,f)

with open('200each/balanced_200each_test_inputs_0.dat','w') as f:
#with open('test_balanced_200each_test_inputs_0.dat','w') as f:
    pkl.dump(test_set_200each_inputs_0,f)

with open('200each/balanced_200each_test_inputs_1.dat','w') as f:
    pkl.dump(test_set_200each_inputs_1,f)

with open('200each/balanced_200each_test_inputs_2.dat','w') as f:
    pkl.dump(test_set_200each_inputs_2,f)

with open('200each/balanced_200each_test_inputs_3.dat','w') as f:
    pkl.dump(test_set_200each_inputs_3,f)

with open('200each/balanced_200each_test_targets.dat','w') as f:
#with open('test_balanced_200each_test_targets.dat','w') as f:
    pkl.dump(test_set_200each_targets,f)
