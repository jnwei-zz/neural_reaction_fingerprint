import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from toolkit import MarkMarkovAtom, GetDoubleBondAtoms, get_molecule_smi
from alkene_rxn_components import Mark_rxn_dict, other_rxn_dict

import numpy as np

rxn_vector_flag = True 
show_prods = True
swap_flag = False 

def write_rxn(mol, nucl, reagents, rxn_vector, prods):
    # @inputs: 
    # mol - rdkit.Mol
    # nucl - rdkit.Mol mol or blank
    # reagents - smiles string
    # prods - list of mols
    #rxn vector = np.array


    if len(prods)==0:
        print 'error, no prods'
        return '' 
    else:
        prod_smi = [Chem.MolToSmiles(prod_mol) for prod_mol in prods[0]]
    
    # Renaming blanks into Nd
    if nucl == '': nucl_smi = '[Nd]'
    else: nucl_smi = get_molecule_smi(nucl)

    if mol == '': mol_smi = '[Nd]'
    else: mol_smi = get_molecule_smi(mol) 

    # adding default solvent:
    if reagents == '':
        reagents = 'COC'
    else:
        reagents += '.COC'

    if show_prods == True:
        rxn_smi = mol_smi + '.' + nucl_smi + '>'+ reagents +'>' + '.'.join(prod_smi) 
    else:
        rxn_smi =  mol_smi + '.' + nucl_smi + '>'+ reagents +'>' 

    if rxn_vector_flag:
        for i in range(len(rxn_vector)): 
            rxn_smi += ',' + str(rxn_vector[i])
    rxn_smi +='\n'

    return rxn_smi


def write_rxn_all_combos_Mark(rxn_mech, mol, marked_mol, other_reactants_list, reagents_list,  rxn_vector):
    # Generating combinations of other reactants and reagents:

    all_rxn_str = ''

    for reag in reagents_list:
    
        # For polymerization reactions
        if rxn_vector[17] == 1: 
            prod_list = rxn_mech.RunReactants((marked_mol,marked_mol))
            all_rxn_str += write_rxn(mol, mol, reag, rxn_vector, prod_list)
            if swap_flag:
                all_rxn_str += write_rxn(mol, mol, reag, rxn_vector, prod_list)

        elif other_reactants_list[0] == '': 
            prod_list = rxn_mech.RunReactants((marked_mol,))
            all_rxn_str += write_rxn(mol, '', reag, rxn_vector, prod_list)
            if swap_flag:
                all_rxn_str += write_rxn('', mol, reag, rxn_vector, prod_list)

        else:
            for o_r in other_reactants_list:
                o_r_mol = Chem.MolFromSmiles(o_r)
                prod_list = rxn_mech.RunReactants((marked_mol,o_r_mol))
                all_rxn_str += write_rxn(mol, o_r_mol, reag, rxn_vector, prod_list)
                if swap_flag:
                    all_rxn_str += write_rxn(o_r_mol, mol, reag, rxn_vector, prod_list)
     
    return all_rxn_str


def write_rxn_all_combos(rxn_mech, mol, other_reactants_list, reagent_list,  rxn_vector):
    # Generating combinations of other reactants and reagents:

    all_rxn_str = ''

    for reag in reagents_list:
        #print other_reactants_list
        if other_reactants_list[0] == '': 
            prod_list = rxn_mech.RunReactants((mol, ))
            if len(prod_list) != 0:
                all_rxn_str += write_rxn(mol, '', reag, rxn_vector, prod_list)
                if swap_flag:
                    all_rxn_str += write_rxn('', mol, reag, rxn_vector, prod_list)
            else:
                print 'alert no products'
        else:  
            for o_r in other_reactants_list:
                o_r_mol = Chem.MolFromSmiles(o_r)
                prod_list = rxn_mech.RunReactants((mol,o_r_mol))
                if len(prod_list) != 0:
                    if rxn_mech == other_rxn_dict['Oz_ox'][0]: 
                        temp_prod_list = tuple([oxidize_aldehyde(prod) for prod in prod_list[0]])
                        prod_list = (temp_prod_list, )
                    all_rxn_str += write_rxn(mol, o_r_mol, reag, rxn_vector, prod_list)
                    if swap_flag:
                        all_rxn_str += write_rxn(o_r_mol, mol, reag, rxn_vector, prod_list)
                else:
                    print 'alert no products'
     
    return all_rxn_str

def oxidize_aldehyde(mol):
    ox_aldehyde = AllChem.ReactionFromSmarts('[C!D3:1]=[O:2]>>[C:1](O)=[O:2]') 
    oxid_prod = ox_aldehyde.RunReactants((mol,))
    if len(oxid_prod) != 0:
        final_prod = oxid_prod[0][0]
    else:
        final_prod = mol
    return final_prod 


def one_hot_encoding(index, vec_length):
    rxn_vec = np.zeros(vec_length)
    rxn_vec[index] = 1.0
    return rxn_vec

### Getting alkenes
with open('alkene_0.txt') as f:
    alkenes = f.readlines()

# Number of random reactions desired in training set 
train_set_size = 10

max_vector_length = 18
target_name_array = ['prob_'+str(i) for i in range(max_vector_length)]
header = 'smiles,'+','.join(target_name_array) + '\n'

for Mark_rxn in Mark_rxn_dict.keys():
    with open('balanced_set/'+Mark_rxn+'.dat', 'w') as f:
        f.write(header)

for rxn in other_rxn_dict.keys():
    with open('balanced_set/'+rxn+'.dat', 'w') as f:
        f.write(header)

debug_flag = False 

for i in xrange(len(alkenes)):

    if i % 100 == 0: print i
    
    alkene_smi = alkenes[i]    
    alk = Chem.MolFromSmiles(alkene_smi)
    alk2 = copy.deepcopy(alk)
    # Determine markovnikov sensitive side.
    # Mark Si side
    double_bond_list = GetDoubleBondAtoms(alk2) 
    at1_id, at2_id = double_bond_list[0]
    Mark_label_alk = MarkMarkovAtom(alk2, at1_id, at2_id)  

    # List of other reactant and reagents
    
    for Mark_rxn in Mark_rxn_dict.keys():
        # React the Mark sensitive side
        if debug_flag: print Mark_rxn
        
        rxn_mech, rxn_num, other_reactants_list, reagents_list = Mark_rxn_dict[Mark_rxn] 
        rxn_vector = one_hot_encoding(rxn_num, max_vector_length)
        
        with open('../balanced_set/'+Mark_rxn+str('.dat'), 'a+') as f:
            f.write(write_rxn_all_combos_Mark(rxn_mech, alk, Mark_label_alk, other_reactants_list, reagents_list, rxn_vector)) 

    for rxn in other_rxn_dict.keys():
        if debug_flag: 
            print rxn

        rxn_mech, rxn_num, other_reactants_list, reagents_list = other_rxn_dict[rxn]  
        rxn_vector = one_hot_encoding(rxn_num, max_vector_length)

        with open('../balanced_set/'+rxn+'.dat', 'a+') as f:
            f.write(write_rxn_all_combos(rxn_mech, alk, other_reactants_list, reagents_list, rxn_vector)) 
