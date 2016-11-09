from os import listdir
from os.path import isfile, join
import autograd.numpy as np

import sys
sys.path.insert(0, '../../neuralfingerprint')
from parse_data import get_train_test_sets, strip_smiles_carrots, split_smiles_triples, load_data_noslice, load_data_noslice 
import pickle as pkl


dat_file_names = [f for f in listdir('../balanced_set') if isfile(join('../balanced_set',f))]
dat_file_names.remove('.keep')
#dat_file_names = ['alkylhal_NR_rxn.dat', 'E_rxn.dat']

num_outputs = 18
target_name_array = ['prob_'+str(i) for i in range(num_outputs)]

for dat_file_nm in dat_file_names: 
    
    inp_smi, targs = load_data_noslice('../balanced_set/'+dat_file_nm, input_name='smiles', 
                target_name_arr=target_name_array)
    
    rxn_ct = len(inp_smi)

    print dat_file_nm +  ' :   ' + str(rxn_ct)
