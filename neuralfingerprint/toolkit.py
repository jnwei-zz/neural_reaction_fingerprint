'''
A library of useful comparisons/measures for reactions

GetDoubleBondAtoms - return a tuple of tuples of atoms in double bonds
CalcBulkiness - requires mol_obj, and atom ids of atoms across double bond
FindBulkierAtom - requires mol_obj, and atom ids of atoms across double bond 
MarkBulkierAtom - returns a mol_obj with bulkier atom replaced with [Si]

edit: 6/1/16:
MarkMarkAtom - returns a mol_obj with Markov preference (i.e. fewer protons)

Bulkiness here compared by: 
  1) number of nonH atoms connected to target atom (on one side of bond)
  2) Sum of atomic mass of neighbors in network

Bulkiness marked by changing the bulkier carbon into a [Si]

# Note: this was copied from neural-fingerprint-reactions/smarts_playground/rxn_framework on 5/27/16 

'''

from rdkit import Chem
from rdkit.Chem import AllChem
from collections import deque


def get_molecule_smi(mol_obj):
    return Chem.MolToSmiles(mol_obj)

def GetDoubleBondAtoms(mol_obj):
    # Returns the pairs of atoms that are part of double bonds as tuples
    patt  = Chem.MolFromSmarts('C=C')
    double_bond_list = mol_obj.GetSubstructMatches(patt)
    return double_bond_list


def GetAdjHalAtoms(mol_obj):
    # Warning, for now only for 2d alkylhalides 
    patt  = Chem.MolFromSmarts('CC([Cl,Br,I])C')
    adj_hal_atoms = mol_obj.GetSubstructMatches(patt)
    # only return geminal carbons
    return (adj_hal_atoms[0][0],  adj_hal_atoms[0][3]) 


def CalcBulkiness(mol_obj, targ_atom_id, pair_atom_id):
    # Calculate the bulkiness away from pair_atom_id specified
    # @pre: targ_atom and pair_atom should both be in the atom 
    # @pre: Limiting scope to radius 3
    radius = 3 

    at1 = mol_obj.GetAtomWithIdx(targ_atom_id)

    # Begin recording neighbors
    at1_tree_list = [thisAtom.GetIdx() for thisAtom in at1.GetNeighbors()]
    myQ = deque(at1_tree_list)

    at1_tree_sum = 0

    while len(myQ) > 0:
        #print at_id
        at_id = myQ.popleft()

        # Don't record if it is one of two starting atoms
        if (at_id == pair_atom_id or at_id == targ_atom_id): 
            at1_tree_list.remove(at_id) 
            continue
        
        curr_at = mol_obj.GetAtomWithIdx(at_id)
        new_neigh_list = [thisAtom.GetIdx() for thisAtom in curr_at.GetNeighbors()]

        for at_id in new_neigh_list:
            if at_id in at1_tree_list: new_neigh_list.remove(at_id)
            
        at1_tree_list.extend(new_neigh_list)
        myQ.extend(new_neigh_list)
    
    for at_idx in at1_tree_list:
        curr_at = mol_obj.GetAtomWithIdx(at_id)
        at1_tree_sum += curr_at.GetMass()

    return at1_tree_list, at1_tree_sum


def FindBulkierAtom(mol_obj, atom1_id, atom2_id):
    # Given a mol_obj, and two atoms (across a double bond), determine which 
    # atom has the bulkier group
    # @pre: atom1_id and atom2_id are across a double bond

    at1_neigh_list, at1_neigh_sum = CalcBulkiness(mol_obj, atom1_id, atom2_id)
    at2_neigh_list, at2_neigh_sum = CalcBulkiness(mol_obj, atom2_id, atom1_id)

    if len(at1_neigh_list) > len(at2_neigh_list):
        return atom1_id
    elif len(at1_neigh_list) == len(at2_neigh_list):
        if at1_neigh_sum >= at2_neigh_sum: return atom1_id

    return atom2_id


def MarkBulkierAtom(mol_obj, atom1_id, atom2_id):
    # Given a molecule with a double bond, mark the bulkier atom
    # with Si 

    bulkyAtom_id = FindBulkierAtom(mol_obj, atom1_id, atom2_id)
    bulkyAtom = mol_obj.GetAtomWithIdx(bulkyAtom_id)
    bulkyAtom.SetAtomicNum(14)

    return mol_obj


def FindMarkAtom(mol_obj, atom1_id, atom2_id):
    # determine which has more H
    at1 = mol_obj.GetAtomWithIdx(atom1_id)
    at2 = mol_obj.GetAtomWithIdx(atom2_id)

    at1_numHs = at1.GetTotalNumHs()
    at2_numHs = at2.GetTotalNumHs()

    if at1_numHs > at2_numHs:
        return atom2_id
    return atom1_id


def MarkMarkovAtom(mol_obj, atom1_id, atom2_id):
    # Given a molecule with a double bond, mark the bulkier atom
    # with Si 

    bulkyAtom_id = FindMarkAtom(mol_obj, atom1_id, atom2_id)
    bulkyAtom = mol_obj.GetAtomWithIdx(bulkyAtom_id)
    bulkyAtom.SetAtomicNum(14)

    return mol_obj


def TestBulkiness(): 
    # Testing the scripts for Bulkiness: 

    test_mol = Chem.MolFromSmiles('C(C)C(C)=CC') 
    bond_list = GetDoubleBondAtoms(test_mol)
    print bond_list
    atom1_id, atom2_id = bond_list[0]

    ## Print Bulkier of two atoms
    #print 'bulkiness for first atom: ', atom1_id
    #print CalcBulkiness(test_mol, atom1_id, atom2_id)

    #print '\nnbulkiness for other atom: ', atom2_id 
    #print CalcBulkiness(test_mol, atom2_id, atom1_id)

    print '\nBulkier atom: ', FindBulkierAtom(test_mol, atom1_id, atom2_id)
    #print '\nBulkier atom: ', FindBulkierAtom(test_mol, atom2_id, atom1_id)

    return 'done'

def testMarkLabel():
    test_mol = Chem.MolFromSmiles('C(C)CC=C(C)C') 
    #test_mol = Chem.MolFromSmiles('C(C)C(C)=CC') 
    patt  = Chem.MolFromSmarts('C=C')
    atom1_id, atom2_id = test_mol.GetSubstructMatches(patt)[0]

    ## Mark the bulkier atom of the double bond
    new_test_mol = MarkMarkovAtom(test_mol, atom1_id, atom2_id)
    print Chem.MolToSmiles(new_test_mol)

if __name__ == '__main__':
    testMarkLabel()

def TestMarkBulkiness():
    # Testing ability to mark bulkiness for Markovnikov orientation
    # Testing some reactions

    test_mol = Chem.MolFromSmiles('C(C)CC=CC') 
    #test_mol = Chem.MolFromSmiles('C(C)C(C)=CC') 
    patt  = Chem.MolFromSmarts('C=C')
    atom1_id, atom2_id = test_mol.GetSubstructMatches(patt)[0]

    ## Mark the bulkier atom of the double bond
    new_test_mol = MarkBulkierAtom(test_mol, atom1_id, atom2_id)
    print Chem.MolToSmiles(new_test_mol)

    ## Match the Si atom
    #patt2 = Chem.MolFromSmarts('[Si]=C')
    #print new_test_mol.HasSubstructMatch(patt2)

    # Test a new reaction:
    hyd_hal_Alkene = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[BrH1,ClH1,IH1:3]>>[C:1]([*:3])[C:2]')
    hyd_hal_Alkene_AntiM = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[BrH1,ClH1,IH1:3]>>[C:2]([*:3])[C:1]')
    HOH_Alkene = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[OH2:3]>>[C:1]([O:3])[C:2]')
    HOH_Alkene_AntiM = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[OH2:3]>>[C:2]([O:3])[C:1]')

    hyd_hal_Alkene = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[Br,Cl,I:3]>>[C:1]([*:3])[C:2]')
    # STEREO : Anti product 
    # Add later if desired
    hal_Alkene = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[Br,Cl,I:3][Br,Cl,I:4]>>[C:1]([*:3])[C:2]([*:4])')

    # STEREO: Anti product 
    # Add later if desired
    epox_Alkene = AllChem.ReactionFromSmarts('[C:1]=[C:2].[C:3][C:4](=[O:5])[O:6][O:7]>>[C:1]1[O:7][C:2]1.[C:3][C:4](=[O:6])[O:5]')
    # Acid required, might react with self is remaining acid is strong enough
    epox_opening = AllChem.ReactionFromSmarts('[C:1]1[O:7][C:2]1>>[C:1]([O:7])[C:2]O') 

    OsO4 = Chem.MolFromSmiles('[Os](O)(O)(O)(O)')
    
    prod_list = hyd_hal_Alkene.RunReactants((new_test_mol, Chem.MolFromSmiles('Br')))
    #prod_list = hyd_hal_Alkene_AntiM.RunReactants((new_test_mol, Chem.MolFromSmiles('[Br][H]')))
    #prod_list = hal_Alkene.RunReactants((new_test_mol, Chem.MolFromSmiles('BrBr')))

    return Chem.MolToSmiles(prod_list[0][0]) 

def TestEpoxideRxn():
    epox_Alkene = AllChem.ReactionFromSmarts('[C:1]=[C:2].[C:3][C:4](=[O:5])[O:6][O:7]>>[C:1]1[O:7][C:2]1.[C:3][C:4](=[O:6])[O:5]')
    # Acid required, might react with self is remaining acid is strong enough
    epox_opening = AllChem.ReactionFromSmarts('[C:1]1[O:7][C:2]1>>[C:1]([O:7])[C:2]O') 
    
    test_mol = Chem.MolFromSmiles('C(C)C(C)=CC') 
    
    patt3 = Chem.MolFromSmarts('CC(=O)OO')
    print Chem.MolFromSmiles('CCC(=O)OO').HasSubstructMatch(patt3)
    print test_mol.HasSubstructMatch(Chem.MolFromSmarts('C=C'))
    
    prod_list = epox_Alkene.RunReactants((test_mol, Chem.MolFromSmiles('CCC(=O)OO')))

    print 'Number of products: ',  len(prod_list[0])
    print  Chem.MolToSmiles(prod_list[0][0])
    print  Chem.MolToSmiles(prod_list[0][1])

    prod_list2 = epox_opening.RunReactants((prod_list[0][0],))
    print 'Number of products of opening: ', len(prod_list2)

    print Chem.MolToSmiles(prod_list2[0][0])

    return 'done'

def TestOsmiumOzoneH2():

    # Osmium oxidzation
    # REQ OsO4
    # STEREO (syn addition) 
    osmium_ox = AllChem.ReactionFromSmarts('[C:1]=[C:2]>>[C:1](O)[C:2](O)')

    # Ozonolysis
    # REQ Ni
    ozon_Alkene = AllChem.ReactionFromSmarts('[C:1]=[C:2]>>[C:1](=O).[C:2](=O)')
    
    # Hydrogenation
    # REQ Pt, Pd, Ni
    # STEREO (syn addition) 
    h2_Alkene_add = AllChem.ReactionFromSmarts('[C:1]=[C:2]>>[C:1][C:2]')

    rxn_dict = {'Os': osmium_ox, 'Oz': ozon_Alkene, 'H2': h2_Alkene_add}
    test_mol = Chem.MolFromSmiles('C(C)C(C)=CC') 

    for rxn_name in rxn_dict.keys():
        print rxn_name
        prod_list = rxn_dict[rxn_name].RunReactants((test_mol, ))
        for prod in prod_list[0]:
            print Chem.MolToSmiles(prod)

    return 'done'

def TestPropOfAlcohol():
    HOH_Alkene_Add = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[OH2:3]>>[C:1]([O:3])[C:2]')
    
    test_mol = Chem.MolFromSmiles('C(C)C(C)=CC') 
    patt  = Chem.MolFromSmarts('C=C')
    atom1_id, atom2_id = test_mol.GetSubstructMatches(patt)[0]
    new_test_mol = MarkBulkierAtom(test_mol, atom1_id, atom2_id)
    
    prod_list = HOH_Alkene_Add.RunReactants((new_test_mol, Chem.MolFromSmiles('O')))
    prod_molH = prod_list[0][0]
    #print Chem.MolToSmiles(prod_mol)
    #prod_mol.UpdatePropertyCache(strict=False)
    #prod_molH = Chem.AddHs(prod_mol)
    prod_molH.UpdatePropertyCache(strict=False)

    Opatt = Chem.MolFromSmarts('O')
    Oatom_id = prod_molH.GetSubstructMatch(Opatt)[0]
    print 'Oxygen atom id: ', Oatom_id
    print 'Charge of oxygen atom: ', prod_molH.GetAtomWithIdx(Oatom_id).GetFormalCharge()
    print 'Degree of oxygen atom: ', prod_molH.GetAtomWithIdx(Oatom_id).GetTotalDegree()
    print 'Bonds of oxygen atom: ', prod_molH.GetAtomWithIdx(Oatom_id).GetBonds()[0].GetEndAtom().GetAtomicNum()


    print 'Valence (Im) of oxygen atom: ', prod_molH.GetAtomWithIdx(Oatom_id).GetImplicitValence()
    print 'Valence (Ex) of oxygen atom: ', prod_molH.GetAtomWithIdx(Oatom_id).GetExplicitValence()
    print 'Num (Im) Hs of oxygen atom: ', prod_molH.GetAtomWithIdx(Oatom_id).GetNumImplicitHs()
    print 'Num (Ex) Hs of oxygen atom: ', prod_molH.GetAtomWithIdx(Oatom_id).GetNumExplicitHs()
    print 'Num of Hs on oxygen atom: ', prod_molH.GetAtomWithIdx(Oatom_id).GetTotalNumHs()

    return 'done'

    


if __name__ == '__main__':
    #print TestBulkiness()
    print TestMarkBulkiness()
    #print TestEpoxideRxn()
    #print TestPropOfAlcohol()
    #print TestOsmiumOzoneH2()


