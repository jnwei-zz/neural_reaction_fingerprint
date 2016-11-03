'''
1/22/16:

1. CreateDoubleBond - Make double bond between any two single bonded carbons
2. AttachRingCarbonBtwnTwoCarbons - Extend carbon rings with another carbon atom.

'''


from rdkit import Chem
from rdkit.Chem import AllChem
#from rdkit.Chem import Draw

def get_molecule_smi(mol_obj):
    return Chem.MolToSmiles(mol_obj)


def AttachCarbonAtCarbon(mol_obj):
    # Find a carbon atom, replace with ethyl group at all possible locations
    # returns a list of molecule objects
    repl = Chem.MolFromSmiles('CC')
    patt = Chem.MolFromSmarts('C')
    return list(AllChem.ReplaceSubstructs(mol_obj, patt, repl))


def AttachChlorineAtCarbon(mol_obj):
    # Attach clorine atom to carbon
    repl = Chem.MolFromSmiles('CCl')
    patt = Chem.MolFromSmarts('C')
    return list(AllChem.ReplaceSubstructs(mol_obj, patt,repl))


def AttachBromineAtCarbon(mol_obj):
    # Attach clorine atom to carbon
    repl = Chem.MolFromSmiles('CBr')
    patt = Chem.MolFromSmarts('C')
    return list(AllChem.ReplaceSubstructs(mol_obj, patt,repl))


def AttachIodineAtCarbon(mol_obj):
    # Attach clorine atom to carbon
    repl = Chem.MolFromSmiles('CI')
    patt = Chem.MolFromSmarts('C')
    return list(AllChem.ReplaceSubstructs(mol_obj, patt,repl))


def CreateDoubleBond(mol_obj):
    # Make a double bond in the molecule.
    rxn = AllChem.ReactionFromSmarts('[C:1][C:2]>>[C:1]=[C:2]')
    #rxnZ = AllChem.ReactionFromSmarts('*[C:1][C:2]*>>*\\[C:1]=[C:2]\\*')
    #rxnE = AllChem.ReactionFromSmarts('*[C:1][C:2]*>>*\\[C:1]=[C:2]/*')
    prod_list = rxn.RunReactants((mol_obj,)) 
    mol_list = []
    for i in range(len(prod_list)): 
        mol_list.append(prod_list[i][0])
    return mol_list


def CheckFeasible(mol_obj):
    # See if you can make a smiles with mol object
    #    if you can't, then skip
    try:
        get_molecule_smi(mol_obj)
    except:
        return False
    return True


def Cleaned(mol_list):
    # Run CheckFeasible() over list of all molecules
    clean_list = [] 
    for mol in mol_list:
        if CheckFeasible(mol): clean_list.append(mol)
    return clean_list


def CheckandRemoveDuplicates(molecule_list):
    # @pre: list is longer than 0 molecules
    if len(molecule_list) < 2:
        return Cleaned(molecule_list) 
    mol_smi_list = [get_molecule_smi(mol) for mol in molecule_list]
    if len(set(mol_smi_list)) < len(mol_smi_list):
        cleaned_smi_list = list(set(mol_smi_list))  
        return Cleaned([Chem.MolFromSmiles(smi) for smi in cleaned_smi_list]) 
    return Cleaned(molecule_list)


def MakeAlkylhalides(start_mol_list, atoms2add = 10, outfname=None):
    # Quick code for generating aliphatic alkylhalides
    # @input start_mol_list - a list with a single molecule to serve as the seed for generating new molecules
    # @input atoms2add - The number of additional carbon atoms you want to add to the chain

    new_mols = start_mol_list 
    for i in xrange(atoms2add):
        #print 'round {}: '.format(i+1)
        temp_new_mols = []
        temp_Cl_mols = []
        temp_Br_mols = []
        temp_I_mols = []
        for start_mol in new_mols:
            temp_new_mols.extend(AttachCarbonAtCarbon(start_mol))
            temp_Cl_mols.extend(AttachChlorineAtCarbon(start_mol))
            temp_Br_mols.extend(AttachBromineAtCarbon(start_mol))
            temp_I_mols.extend(AttachIodineAtCarbon(start_mol))
        #print type(temp_new_mols[0])
        temp_new_mols = CheckandRemoveDuplicates(temp_new_mols)
        temp_Cl_mols = CheckandRemoveDuplicates(temp_Cl_mols)
        temp_Br_mols = CheckandRemoveDuplicates(temp_Br_mols)
        temp_I_mols = CheckandRemoveDuplicates(temp_I_mols)
        
        new_mols = temp_new_mols
        
        if outfname is not None:
            with open(outfname, 'a') as f:
                for mol_list in [temp_Cl_mols, temp_Br_mols, temp_I_mols]:
                    for mol in mol_list:
                        f.write(get_molecule_smi(mol) + '\n')
        else:
            for mol_list in [temp_Cl_mols, temp_Br_mols, temp_I_mols]:
                for mol in mol_list:
                    print get_molecule_smi(mol)
    
    return 'done'


def MakeAlkenes(start_mol_list, atoms2add = 10, outfname = None):
    # Quick code for generating alkylhalides
    # @input start_mol_list - a list with a single molecule to serve as the seed for generating new molecules
    # @input atoms2add - The number of additional carbon atoms you want to add to the chain

    new_mols = start_mol_list 
    for i in xrange(atoms2add):
        temp_new_mols = []
        temp_alkenes = []
        for start_mol in new_mols:
            temp_new_mols.extend(AttachCarbonAtCarbon(start_mol))
            temp_alkenes.extend(CreateDoubleBond(start_mol))
        temp_new_mols = CheckandRemoveDuplicates(temp_new_mols)
        temp_alkenes = CheckandRemoveDuplicates(temp_alkenes)
        
        new_mols = temp_new_mols
        
        if outfname is not None:
            with open(outfname, 'a') as f:
                for mol in temp_alkenes:
                    f.write(get_molecule_smi(mol) + '\n')
        else:
            for mol in temp_alkenes:
                print get_molecule_smi(mol)

    return 'done'


if __name__ == '__main__':
    # Starting molecule:
    start_mol_list = [Chem.MolFromSmiles('C')]

    MakeAlkylhalides(start_mol_list, atoms2add = 10, outfname = 'alkylhalides.dat')
    MakeAlkenes(start_mol_list, atoms2add = 10, outfname = 'alkenes_0.txt')
