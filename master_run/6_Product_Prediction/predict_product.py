''''
Predict the product and compare with the answer

Mostly for exam questions, for my generated questions should be the same
Based upon the results of the prediction vector.

TO Consider: Decide how to handle markovnikov reactions
    - Either: Label alkenes as before to get right markovnikov side
        : This ends up providing alg. with extra info to help get product right, do we want it to be that successful?
    - Or: Remove Si label from reaction smarts, and leave up to probability
'''
import pickle as pkl
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

import copy
from rdkit.Chem import AllChem
from rxns_lib import Full_rxn_dict
from neuralfingerprint.parse_data import split_smiles_triples
from neuralfingerprint.toolkit import MarkMarkovAtom, GetDoubleBondAtoms, get_molecule_smi


def returnRxnObjfromPred(rxn_num):
    # helper function to parse dictionary for reaction type.
    # rxn_num is a string, not '0' (NR separtely handled)
    rxn_obj = Full_rxn_dict[str(rxn_num)] 
    return rxn_obj

def returnProductsfromRxnObj(rxn_obj, rct1_smi, rct2_smi):
    # run reaction from the rct smi, generate the products molecules
    rct1 = Chem.MolFromSmiles(rct1_smi)
    if rct2_smi == '[Nd]':
        prods = rxn_obj.RunReactants((rct1,))
    else:
        rct2 = Chem.MolFromSmiles(rct2_smi)
        prods = rxn_obj.RunReactants((rct1,rct2))

    if len(prods) != 0:
        prod_smi_list =  [Chem.MolToSmiles(prod_mol) for prod_mol in prods[0]]
    else:
        if rct2_smi == '[Nd]' : prod_smi_list = [rct1_smi]
        else: prod_smi_list = [rct1_smi, rct2_smi]
    return '.'.join(prod_smi_list) 

def returnProductsfromMarkRxnObj(rxn_obj, rct1_smi, rct2_smi):
    # Run reaction using Mark labelled reaction options 
    alk = Chem.MolFromSmiles(rct1_smi)

    # Labeling mark Alkene:
    alk2 = copy.deepcopy(alk)
    double_bond_list = GetDoubleBondAtoms(alk2) 
    at1_id, at2_id = double_bond_list[0]
    Mark_label_alk = MarkMarkovAtom(alk2, at1_id, at2_id)  

    if rct2_smi != '[Nd]':
        rct2 = Chem.MolFromSmiles(rct2_smi)
        prods = rxn_obj.RunReactants((Mark_label_alk,rct2))
    else:
        try:
            prods = rxn_obj.RunReactants((Mark_label_alk,))
        except:
            print 'invalid number of reactants'
            prods = [] 

    if len(prods) != 0:
        prod_smi_list =  [Chem.MolToSmiles(prod_mol) for prod_mol in prods[0]]
    else:
        if rct2_smi == '[Nd]' : prod_smi_list = [rct1_smi]
        else: prod_smi_list = [rct1_smi, rct2_smi]
    return '.'.join(prod_smi_list) 

def returnProductsfromPred(rxn_num, rct1_smi, rct2_smi):
    # Wrapper function for writing out products given various options 
    # @output : a string of all the products (with the dots)

    # NR
    if rxn_num == '0': return '[Nd]'

    # Markovnikov reactions
    # List of Markovnikov reactions
    Mark_list = ['5', '6', '7', '8', '9', '12', '17'] 
    if rxn_num in Mark_list:
        rxn_obj = returnRxnObjfromPred(rxn_num)
        return returnProductsfromMarkRxnObj(rxn_obj, rct1_smi, rct2_smi)

    # Everything else
    return returnProductsfromRxnObj(returnRxnObjfromPred(rxn_num), rct1_smi, rct2_smi)

def tanimotoComparison(pred_prod_list, true_prod_list):
    # Return the tanimoto score
    pred_mol = Chem.MolFromSmiles(pred_prod_list)
    answer_mol = Chem.MolFromSmiles(true_prod_list)

    pred_fps = FingerprintMols.FingerprintMol(pred_mol) 
    answer_fps = FingerprintMols.FingerprintMol(answer_mol) 
    
    return DataStructs.FingerprintSimilarity(pred_fps, answer_fps)


if __name__== '__main__':
    # Get predictions, calculated elsewhere
    vec_pred = pkl.load(open('../results/class_3_1_neural1_200each_Wade_prob8_47.dat')) 
    
    with open('../../data/test_question/prob8_47.cf.txt') as probf:
        rxn_smis = probf.readlines() 

    with open('../../data/test_question/prob8_47.ans_smi.txt') as ansf:
        ans_rxn_smis = ansf.readlines()

    test_input_1, _, _ = split_smiles_triples(rxn_smis)

    for ii in range(np.shape(vec_pred)[0]):
        pred_type = np.argmax( vec_pred[ii,:])
        #print pred_type

        # get reactants for product prediction
        rct1_smi, rct2_smi, _ = test_input_1[ii]
        prods = returnProductsfromPred(str(pred_type), rct1_smi, rct2_smi)
        print ans_rxn_smis[ii], prods

        print 'Similarity score: ', tanimotoComparison(prods, ans_rxn_smis[ii])
        print '\n'

