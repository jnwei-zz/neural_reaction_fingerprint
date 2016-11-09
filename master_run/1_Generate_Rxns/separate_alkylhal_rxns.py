'''
Draft 2 of a code to generate reaction smiles
    - Based on paper chemistry rules for Alkanes
    - For now, assume low heat

Additions from draft 1:
    - Allow for order of reactants to swap
    - Prints out vector for likelihood of products


To Do:
    - Zaitsev rules?  -- if care about product shape 
    - bulky categorization (not just geminal to tertiary group, but also large secondary groups)

    - Markovnikov orientation:
            Mark the carbon that receives heavy group with C13.
'''

from rdkit import Chem
from rdkit.Chem import AllChem
#from substrate_gen import AttachCarbonAtCarbon, AttachChlorineAtCarbon,\
#            AttachBromineAtCarbon, AttachIodineAtCarbon,\
#            CheckandRemoveDuplicates, get_molecule_smi
from generate_substrates import get_molecule_smi
from toolkit import GetAdjHalAtoms, MarkBulkierAtom
import numpy as np
import copy

##############
### Reaction conditions
##############
# 
# Includes: solvent, nucleophiles
#     To add: temperature
#

## nucleophile/base dict
nucl_dict = {}
nucl_dict['cyanide'] = Chem.MolFromSmiles('[C-]#N')
nucl_dict['hydroxide'] = Chem.MolFromSmiles('[OH-]')
nucl_dict['tertbutyl-O'] = Chem.MolFromSmiles('[O-]C(C)(C)C')
nucl_dict['water'] = Chem.MolFromSmiles('O')


## solvents:
#solvent_dict = {'water': 'O', 'acetone':'CC(=O)C'}#, 'diethyl_ether':'CCOCC', 'hexane':'CCCCCC'}

num_rxn_types = 18
swap = False 
show_prods = True
show_rxn_vec_str = True 


###############
#### Reaction SMARTS
###############
# Still need to consider:
#       - Zaitsev's rule


#####################
######ALKYLHALIDE REACTIONS
## SN reaction:
sn2_rxn = AllChem.ReactionFromSmarts('[C:1][Br,Cl,I:2].[C,O,N,S-:3]>>[C:1][*-0:3].[*-:2]')
sn_methyl_shift_rxn =  AllChem.ReactionFromSmarts('[C:1][CD4:2][C:3][Br,Cl,I:4].[O,N,S:5]>>[CD4:2]([*:5])[C:3][C:1].[*:4]')

## E reaction:
#E_rxn = AllChem.ReactionFromSmarts('[CH1,CH2,CH3:1]-[C:2][Br,Cl,I:3].[OH0-,OH-:4]>>[C:1]=[C:2].[OH2-0:4].[*-:3]')
E_rxn = AllChem.ReactionFromSmarts('[CH1,CH2,CH3:1]-[C:2][Br,Cl,I:3].[O-:4]>>[C:1]=[C:2].[O-0:4].[*-:3]')
E_rxn_OH = AllChem.ReactionFromSmarts('[CH1,CH2,CH3:1]-[C:2][Br,Cl,I:3].[OH-:4]>>[C:1]=[C:2].[OH2-0:4].[*-:3]')
E_Zait_rxn = AllChem.ReactionFromSmarts('[SiH1,SiH2,SiH3:1]-[C:2][Br,Cl,I:3].[O-:4]>>[C:1]=[C:2].[O-0:4].[*-:3]')
E_methyl_shift_rxn = AllChem.ReactionFromSmarts('[C:1][CD4:2][C:3][Br,Cl,I:4].[O,N,S:5]>>[C:2]=[C:3][C:1].[*+1:5].[*-:4]')

target_name_array = ['prob_'+str(i) for i in range(num_rxn_types)]
header = 'smiles,'+','.join(target_name_array) + '\n'

for fname in ['../balanced_set/sn2_rxn.dat','../balanced_set/E_rxn.dat', 
        '../balanced_set/alkylhal_NR_rxn.dat', '../balanced_set/sn2m_rxn.dat','../balanced_set/Em_rxn.dat']: 
    with open(fname,'w') as f:
        f.write(header)

###########
# Identify type of alkylhalide substrate
###########
#
# primary, secondary, tertiary, methylhalide; also bulky or not bulky 
#

match_dict = {}
match_dict['methylhalide'] = Chem.MolFromSmarts('[CD1]-[Br,Cl,I]')
match_dict['primary'] = Chem.MolFromSmarts('[CD2]-[Br,Cl,I]')  
match_dict['secondary'] = Chem.MolFromSmarts('[CD3]-[Br,Cl,I]')  
match_dict['tertiary'] = Chem.MolFromSmarts('[CD4]-[Br,Cl,I]')  
#match_dict['bulky'] = Chem.MolFromSmarts('C[CD4]C[Br,Cl,I]')   
# Do I really need the extra carbon?
match_dict['bulky'] = Chem.MolFromSmarts('C[CD4]C[Br,Cl,I]')   
match_dict['double_bulky'] = Chem.MolFromSmarts('[CD4]C([Br,Cl,I])[CD4]')   

def test_match(mol, match_smarts_key):
    # Return a whether a substructure is matched in a target molecule
    return mol.HasSubstructMatch(match_dict[match_smarts_key])


def Mark_Gem_Carbon(mol):
    adj_hal_tup = GetAdjHalAtoms(mol)
    at1_id, at2_id = adj_hal_tup
    Mark_label_alkhal = MarkBulkierAtom(mol, at1_id, at2_id)
    return Mark_label_alkhal


def classify_substrate(mol):
    if test_match(mol, 'methylhalide'): return 'methylhalide'
    if test_match(mol, 'primary'):
        if test_match(mol, 'bulky'): return 'primary_bulky'
        else: return 'primary'
    if test_match(mol, 'secondary'):
        if test_match(mol, 'double_bulky'): return 'secondary_double_bulky'
        if test_match(mol, 'bulky'): return 'secondary_bulky'
        else: return 'secondary'
    if test_match(mol, 'tertiary'): return 'tertiary'
    return 'not classified'

#############
### Write reactions based on substrate and nucleophile
#############
#
# Modularized for easy reading.
# return the reaction smiles directly


# This version will write vectors for the reaction products

def write_rxn(mol, nucl, solv, rxn_vec_str, prods):
    # Writes rxn smiles based on rxn_vec is a string of num_outputs csv
    
    prod_smi = [Chem.MolToSmiles(prod_mol) for prod_mol in prods[0]]

    rxn_smi = get_molecule_smi(mol) + '.' + get_molecule_smi(nucl) + '>'+ solv +'>' 
    if show_prods == True: 
        rxn_smi += '.'.join(prod_smi)
    if show_rxn_vec_str:
        rxn_smi += ',' + rxn_vec_str
    
    rxn_smi += '\n'
    if swap == True:
        rxn_smi +=   get_molecule_smi(nucl) + '.' + get_molecule_smi(mol) + '>'+ solv +'>' 
        if show_prods == True: 
            rxn_smi += '.'.join(prod_smi)  
        if show_rxn_vec_str:
            rxn_smi += ',' + rxn_vec_str

    return rxn_smi

def write_rxn_NR(mol, nucl, solv):
    NR_vec = np.zeros(num_rxn_types-1)
    rxn_vec_str = '1.0'
    
    for i in range(len(NR_vec)): 
        rxn_vec_str += ',' + str(NR_vec[i])

    rxn_smi = get_molecule_smi(mol) + '.' + get_molecule_smi(nucl) + '>'+ solv +'>' 

    if show_prods == True: 
        rxn_smi += '[Nd]' 
    if show_rxn_vec_str:
        rxn_smi += ','+rxn_vec_str

    rxn_smi += '\n'
    if swap == True:
        rxn_smi +=  get_molecule_smi(nucl) + '.' + get_molecule_smi(mol) + '>'+ solv +'>'
        if show_prods == True: 
            rxn_smi += '[Nd]' 
        if show_rxn_vec_str:
            rxn_smi += ',' + rxn_vec_str

    return rxn_smi


def write_rxn_vec_str(rxn_dict):
    # keys of rxn_dict are the rxn nums, arguments are the probabilities
    #   of that reaction type

    # First make a 1 x num_rxn_types vector for all vectors 
    rxn_vec = np.zeros(num_rxn_types)
    if len(rxn_dict.keys()) == 1:
        rxn_vec[int(rxn_dict.keys()[0])] = 1.0
    else:
        for kwarg in rxn_dict.keys():
            rxn_vec[int(kwarg)] = rxn_dict[kwarg]

    # Sanity check 
    assert np.sum(rxn_vec) == 1

    # writing the string
    rxn_vec_str = ''
    for idx in range(len(rxn_vec)):
        if idx != len(rxn_vec)-1:
            rxn_vec_str += str(rxn_vec[idx]) + ',' 
        else:
            rxn_vec_str += str(rxn_vec[idx]) 

    return rxn_vec_str 

'''

############
# Original version
# This version writes the full reaction smiles 

def write_rxn(mol, nucl, solv, rxn_num, prods):
    no_solv = True
    rxn_num_label = True

    prod_smi = [Chem.MolToSmiles(prod_mol) for prod_mol in prods[0]]
    rxn_smi = get_molecule_smi(mol) + '.' + get_molecule_smi(nucl) + '>'+ solv +'>' + '.'.join(prod_smi)

    if no_solv and rxn_num_label:
        rxn_smi = get_molecule_smi(mol) + '.' + get_molecule_smi(nucl) + '>>' + '.'.join(prod_smi)+','+ str(rxn_num)
        #rxn_smi = get_molecule_smi(mol) + '.' + get_molecule_smi(nucl) + '>>' + '.'.join(prod_smi)  
    return rxn_smi

def write_rxn_NR(mol, nucl, solv):
    no_solv = True
    rxn_num_label = True

    rxn_smi = get_molecule_smi(mol) + '.' + get_molecule_smi(nucl) + '>'+ solv +'>NR'
    if no_solv and rxn_num_label:
        rxn_smi = get_molecule_smi(mol) + '.' + get_molecule_smi(nucl) + '>>NR,0'
    return rxn_smi 
'''

def tertiary_rxn_alg(mol,nucl, solv):
    # Only dependent on nucleohpile
    #   and heat, but that can come later
    # Also need to adjust for methyl/hydride shifts... may be I'll add this later 
    if nucl == nucl_dict['cyanide']:
        #print 'Mechanism SN1'
        prods = sn2_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'1':1.0})
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    elif nucl == nucl_dict['tertbutyl-O']:
        #print 'Mechanism E2'
        prods = E_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'2':1.0})
        with open('../balanced_set/E_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0 
    elif nucl == nucl_dict['hydroxide']: 
        prods = E_rxn_OH.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'2':1.0})
        with open('../balanced_set/E_rxn.dat','a') as f:  # no diff for mech between E and E_rxn_OH
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return  0
    elif nucl == nucl_dict['water']:
        with open('../balanced_set/alkylhal_NR_rxn.dat','a') as f: 
            f.write(write_rxn_NR(mol, nucl, solv))
        return 0 
    return 'nucleophile not identified'


def methylhalide_rxn_alg(mol, nucl, solv):
    if nucl == nucl_dict['cyanide']:
        #print 'Mechanism SN2'
        prods = sn2_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'1':1.0})
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    elif nucl == nucl_dict['hydroxide']:
        #print 'Mechanism SN2'
        prods = sn2_rxn.RunReactants((mol, nucl))
        this_rxn_vec_str = write_rxn_vec_str({'1':1.0})
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    elif nucl == nucl_dict['tertbutyl-O']:
        #print 'no reaction' 
        return write_rxn_NR(mol, nucl, solv)
    elif nucl == nucl_dict['water']:
        with open('../balanced_set/alkylhal_NR_rxn.dat','a') as f: 
            f.write(write_rxn_NR(mol, nucl, solv))
        return 0 
    return 'nucleophile not identified'


def primary_rxn_alg(mol, nucl, solv):
    if nucl == nucl_dict['cyanide']:
        #print 'Mechanism SN2'
        prods = sn2_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'1':1.0})
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    elif nucl == nucl_dict['hydroxide']:
        #print 'Mechanism SN2 + E2'
        prods_1 = sn2_rxn.RunReactants((mol, nucl))
        prods_2 = E_rxn_OH.RunReactants((mol,nucl))
        #print (prods_1[0] + prods_2[0], )
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv,  write_rxn_vec_str({'1':0.5, '2':0.5}), (prods_1[0]+prods_2[0], ))) 
        with open('../balanced_set/E_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv,  write_rxn_vec_str({'1':0.5, '2':0.5}), (prods_1[0]+prods_2[0], ))) 
        return 0 
    elif nucl == nucl_dict['tertbutyl-O']:
        #print 'Mechanism E2'
        prods = E_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'2':1.0})
        return write_rxn(mol, nucl, solv, this_rxn_vec_str, prods)
    elif nucl == nucl_dict['water']:
        #print 'no reaction'
        with open('../balanced_set/alkylhal_NR_rxn.dat','a') as f: 
            f.write(write_rxn_NR(mol, nucl, solv))
        return 0 
    return 'nucleophile not identified'

def primary_rxn_bulky_alg(mol, nucl, solv):
    # Cannot undergo elimination
    if nucl == nucl_dict['cyanide']:
        #print 'Mechanism SN2'
        prods = sn2_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'1':1.0})
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    if nucl == nucl_dict['hydroxide']:
        #print 'Mechanism SN2'
        prods = sn2_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'1':1.0})
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    elif nucl == nucl_dict['tertbutyl-O']:
        #print 'no reaction'
        with open('../balanced_set/alkylhal_NR_rxn.dat','a') as f: 
            f.write(write_rxn_NR(mol, nucl, solv))
        return 0 
    elif nucl == nucl_dict['water']:
        #print 'no reaction'
        with open('../balanced_set/alkylhal_NR_rxn.dat','a') as f: 
            f.write(write_rxn_NR(mol, nucl, solv))
        return 0 
    return 'nucleophile not identified'

def secondary_rxn_alg(mol, nucl, solv):
    if nucl == nucl_dict['cyanide']:
        #print 'Mechanism SN2'
        prods = sn2_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'1':1.0})
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    elif nucl == nucl_dict['hydroxide']:
        #print 'Mechanism SN2'
        prods = sn2_rxn.RunReactants((mol, nucl))
        #return write_rxn(mol, nucl, solv, 1, prods)
        this_rxn_vec_str = write_rxn_vec_str({'1':1.0})
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    elif nucl == nucl_dict['tertbutyl-O']:
        # for labeling the product
        mol2 = copy.deepcopy(mol) 
        Mark_labeled_alkhal = Mark_Gem_Carbon(mol2)
        #print 'Mechanism E2'
        prods = E_Zait_rxn.RunReactants((Mark_labeled_alkhal,nucl))
        #return write_rxn(mol, nucl, solv, 2, prods)
        this_rxn_vec_str = write_rxn_vec_str({'2':1.0})
        with open('../balanced_set/E_rxn.dat','a') as f:  # no diff for mech between E and E_rxn_OH
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return  0
    elif nucl == nucl_dict['water']:
        #print 'no reaction'
        with open('../balanced_set/alkylhal_NR_rxn.dat','a') as f: 
            f.write(write_rxn_NR(mol, nucl, solv))
        return 0 
    return 'nucleophile not identified'

def secondary_rxn_bulky_alg(mol, nucl, solv):
    #print 'secondary and bulky'
    # should account for methyl hydride shifts if water is nucleophile 
    if nucl  == nucl_dict['cyanide']:
        #print 'Mechanism SN2'
        prods = sn2_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'1':1.0})
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    elif nucl == nucl_dict['hydroxide']:
        # May be E2 at a higher temp...
        #print 'Mechanism SN2'
        prods = sn2_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'1':1.0})
        with open('../balanced_set/sn2_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    elif nucl == nucl_dict['tertbutyl-O']:
        #print 'Mechanism E2'
        prods = E_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'2':1.0})
        with open('../balanced_set/E_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0 
    elif nucl == nucl_dict['water']:
        #print 'SN2 and E2 methyl hydride shift' 
        prods_1 = sn_methyl_shift_rxn.RunReactants((mol,nucl))
        prods_2 = E_methyl_shift_rxn.RunReactants((mol,nucl))
        with open('../balanced_set/sn2m_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv,  write_rxn_vec_str({'3':0.5, '4':0.5}), (prods_1[0]+prods_2[0], ))) 
        with open('../balanced_set/Em_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv,  write_rxn_vec_str({'3':0.5, '4':0.5}), (prods_1[0]+prods_2[0], ))) 
        return 0 
    return 'nucleophile not identified'

def secondary_rxn_double_bulky_alg(mol, nucl, solv):
    # Handles fact that reactions that E1 reaction can't occur when doubly bulky
    if nucl == nucl_dict['water']:
        prods = sn_methyl_shift_rxn.RunReactants((mol,nucl))
        this_rxn_vec_str = write_rxn_vec_str({'3':1.0})
        with open('../balanced_set/sn2m_rxn.dat','a') as f:
            f.write(write_rxn(mol, nucl, solv, this_rxn_vec_str, prods))
        return 0
    elif nucl == nucl_dict['tertbutyl-O']:
        with open('../balanced_set/alkylhal_NR_rxn.dat','a') as f: 
            f.write(write_rxn_NR(mol, nucl, solv))
        return 0 
    else:
        return secondary_rxn_bulky_alg(mol, nucl, solv)
        

###########
##### Main code starts here:
##### Generating reactions for all substrates:
###########
rxn_dict = {'methylhalide':methylhalide_rxn_alg, 'primary': primary_rxn_alg, 'primary_bulky':  primary_rxn_bulky_alg, 'tertiary': tertiary_rxn_alg, 'secondary': secondary_rxn_alg, 'secondary_bulky':  secondary_rxn_bulky_alg, 'secondary_double_bulky': secondary_rxn_double_bulky_alg}

with open('alkylhalides.dat') as f:
    master_list = f.readlines()

#test_mol_list = ['CBr', 'CCBr', 'CC(C)(C)CBr', 'CC(Br)C', 'CC(C)(C)C(Br)C', 'CC(C)(Br)C', 'CC(C)(C)C(Br)C(C)(C)C']

#newf = open('03_22_rxns/alkylhal_rxn_swap_vec.dat','w')


# change files to be in dictionary form
# add to files

#for mol_smi in test_mol_list:
for mol_smi in master_list:
    #newf.write(get_molecule_smi(mol)+'\n')
    for nucl_name in nucl_dict.keys():
        nucl = nucl_dict[nucl_name]
        mol = Chem.MolFromSmiles(mol_smi) 
        solv = 'COC' 
        type_str = classify_substrate(mol)
        rxn_dict[type_str](mol, nucl, solv)

##############################
# Not used for now 
###############################

#### Main reaction:
## first test on this set of reactants:
#test_mol_list = ['CBr', 'CCBr', 'CC(C)(C)CBr', 'CC(Br)C', 'CC(C)(C)C(Br)C', 'CC(C)(Br)C']
#
## Dictionary for which algorithm to send the reaction to
#
#for mol_smi in test_mol_list:
#    print '\n', mol_smi
#    mol = Chem.MolFromSmiles(mol_smi)
#    for nucl_name in nucl_dict.keys():
#        nucl = nucl_dict[nucl_name]
#        solv = solvent_dict['acetone']
#        type_str = classify_substrate(mol) 
#        if type_str != 'methylhalide':
#            print rxn_dict[type_str](mol, nucl, solv) 


