import pickle as pkl
from rdkit import Chem, DataStructs
import numpy as np
from rdkit.Chem.Fingerprints import FingerprintMols
import sys

sys.path.append('../../')


# change this to location of training input directory
train_input_0 = pkl.load(open("data/200each/balanced_200each_train_inputs_0.dat"))
    
problem_list = ['Wade8_47', 'Wade8_48']

with open('type1_questions/'+ problem_list[0]+'.cf.txt') as f1:
    input_smiles = f1.readlines()

with open('answers.txt') as ansf:
   answers = ansf.readlines() 

match = 0

ave_tanimoto = np.zeros(len(input_smiles))
high_tanimoto = np.zeros(len(input_smiles))
high_sim_smi = ['' for ii in range(len(input_smiles))]

for smi_idx, test_smi in enumerate(input_smiles):
    test_smi = test_smi.rstrip()
    test_smi = test_smi.replace('>','.')
    while test_smi[-1] ==  '.': 
        test_smi = test_smi[:-1]
    print test_smi
    test_smi_fp = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(test_smi)) 
    temp_sum_tanimoto_dist = 0.0
    for train_smi in train_input_0:
        train_smi_parts = train_smi.split('.')[:-1]
        new_train_smi = '.'.join(train_smi_parts) 
        if test_smi in new_train_smi:
            print 'found a match'
            print 'test smiles: ', test_smi
            print 'train smiles: ', new_train_smi
            match += 1
        train_smi_fp = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(new_train_smi)) 
        tanimoto_dist = DataStructs.FingerprintSimilarity(test_smi_fp, train_smi_fp)
        temp_sum_tanimoto_dist += tanimoto_dist 
        if tanimoto_dist > high_tanimoto[smi_idx]: 
            high_tanimoto[smi_idx] = tanimoto_dist
            high_sim_smi[smi_idx] = new_train_smi
    ave_tanimoto[smi_idx] = temp_sum_tanimoto_dist/ len(train_input_0)

print 'num of matches = ', match 
print 'average tanimoto score of training set versus test questions'
print ave_tanimoto

print 'highest tanimoto score within training set versus test questions'
print high_tanimoto
print high_sim_smi
