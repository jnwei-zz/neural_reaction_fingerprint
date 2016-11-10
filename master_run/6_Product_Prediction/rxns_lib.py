'''
A collection of reactions for use in generating the products
These SMARTS are identical to those in the generation document, just compiled together here for convenience.
'''

from rdkit.Chem import AllChem


##########################
### Alkylhalide reactions
##########################

# SN reactions (as copied from alkylhal_rxns.py 5/26/16)
sn2_rxn = AllChem.ReactionFromSmarts('[C:1][Br,Cl,I:2].[C,O,N,S-:3]>>[C:1][*-0:3].[*-:2]')
sn_methyl_shift_rxn =  AllChem.ReactionFromSmarts('[C:1][CD4:2][C:3][Br,Cl,I:4].[O,N,S:5]>>[CD4:2]([*:5])[C:3][C:1].[*:4]')

# E reactions (as copied from alkylhal_rxns.py 5/26/16)
E_rxn = AllChem.ReactionFromSmarts('[CH1,CH2,CH3:1]-[C:2][Br,Cl,I:3].[O-:4]>>[C:1]=[C:2].[O-0:4].[*-:3]')
E_methyl_shift_rxn = AllChem.ReactionFromSmarts('[C:1][CD4:2][C:3][Br,Cl,I:4].[O,N,S:5]>>[C:2]=[C:3][C:1].[*+1:5].[*-:4]')

# These reactions were used for generating product, did not predict on them...
#E_rxn_OH = AllChem.ReactionFromSmarts('[CH1,CH2,CH3:1]-[C:2][Br,Cl,I:3].[OD0-:4]>>[C:1]=[C:2].[OH2-0:4].[*-:3]')
#E_Zait_rxn = AllChem.ReactionFromSmarts('[SiH1,SiH2,SiH3:1]-[C:2][Br,Cl,I:3].[O-:4]>>[C:1]=[C:2].[O-0:4].[*-:3]')

##########################
### Alkene reactions
##########################

hyd_hal_Alkene = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[Br,Cl,I:3]>>[C:1]([*:3])[C:2]')
hyd_hal_Alkene_AntiM = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[Br:3]>>[C:2]([Br:3])[C:1]')
HOH_Alkene = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[OH2:3]>>[C:1]([O:3])[C:2]')
HOH_Alkene_AntiM = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[OH2:3]>>[C:2]([O:3])[C:1]')

ROH_Alkene = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[C:4][O:3]>>[C:1]([O:3][C:4])[C:2]')
#ROH_Alkene_AntiM = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[C:4][O:3]>>[C:2]([O:3][C:4])[C:1]')

# STEREO : Anti product 
# Add later if desired
### !!! lookout, should technically be 3 reactants
halhyd_Alkene = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[Br,Cl,I:3][Br,Cl,I:4]>>[C:1](O)[C:2]([*:3]).[*-:4]')
    
# STEREO : Anti product 
# Add later if desired
halhal_Alkene = AllChem.ReactionFromSmarts('[C:1]=[C:2].[Br,Cl,I:3][Br,Cl,I:4]>>[C:1]([*:3])[C:2]([*:4])')

# STEREO: Anti product 
# Add later if desired
epox_Alkene = AllChem.ReactionFromSmarts('[C:1]=[C:2].[C:4](=[O:5])[O:6][O:7]>>[C:1]1[O:7][C:2]1.[C:4](=[O:6])[O:5]')
# Acid required, might react with self is remaining acid is strong enough
epox_opening = AllChem.ReactionFromSmarts('[C:1]1[O:7][C:2]1>>[C:1]([O:7])[C:2]O') 
    
# Osmium oxidzation
# REQ OsO4
# STEREO (syn addition) 
oxid_Alkene = AllChem.ReactionFromSmarts('[C:1]=[C:2]>>[C:1](O)[C:2](O)')

# Ozonolysis
ozon_ox_Alkene = AllChem.ReactionFromSmarts('[C:1]=[C:2].O~O~O>>[C:1](=O).[C:2](=O)')     #Match products accoridngly
ozon_red_Alkene = AllChem.ReactionFromSmarts('[C:1]=[C:2].O~O~O>>[C:1](=O).[C:2](=O)')

# Hydrogenation
# REQ Pt, Pd, Ni
# STEREO (syn addition) 
h2_Alkene_add = AllChem.ReactionFromSmarts('[C:1]=[C:2].[H][H]>>[C:1][C:2]')
polymerization = AllChem.ReactionFromSmarts('[Si:1]=[C:2].[Si:3]=[C:4]>>O[C:1][C:2][C:3][C:4]')

# For now, reaction dictionaries are written in tuples of the following format:
#   1. Reaction name (defined above)
#   2. Index number
#   3. Listed other reactants
#   4. Other required reagents (structure is not attached to main product molecule).
Mark_rxn_dict = {
    'hyd_hal':          (hyd_hal_Alkene, 5, ['Br', 'Cl', 'I'],[ '']),
    'hyd_hal_AntiM':    (hyd_hal_Alkene_AntiM, 6, ['Br'], ['CCOOCC','CC(=O)OOC(=O)C']),
    'HOH':              (HOH_Alkene, 7, ['O'], ['OS(O)(=O)=O','CC(=O)O[Hg]OC(C)=O.O.[Na+].[H][B-]([H])([H])[H]']), 
    'HOH_AntiM':        (HOH_Alkene_AntiM, 8, ['O'], [ '[H]B([H])[H].C1CCOC1.OO.[Na+].[O-]', '[H]B([H])[H].C1CCOC1.OO.[Na+].C(C)(C)C[O-]']),
    'ROH':              (ROH_Alkene, 9, ['CO', 'CCO', 'CCCO'], ['CC(=O)O[Hg]OC(C)=O.O.[Na+].[H][B-]([H])([H])[H]']),  
    #'ROH_AntiM':        (ROH_Alkene_AntiM, 10, ['CO', 'CCO', 'CCCO'], ['[H]B([H])[H].C1CCOC1.OO.[K+].[O-]', 
    #        '[H]B([H])[H].C1CCOC1.OO.[Na+].[O]', '[H]B([H])[H].C1CCOC1.OO.[Na+].C(C)(C)C[O-]']),
    'halhyd':           (halhyd_Alkene, 12, ['BrBr','ClCl', 'II'], ['O']),
    'polymer':  (polymerization, 17, [''], ['OS(O)(=O)=O.O'])
    }
other_rxn_dict = {
    'halhal':   (halhal_Alkene, 11, ['BrBr','ClCl', 'II'], ['','C([Cl])([Cl])']),#'C([Cl])([Cl])([Cl])', 'C([Cl])([Cl])([Cl])([Cl])']), 
    'epox':     (epox_Alkene, 13,  ['CC(=O)OO', 'CCC(=O)OO', 'CCCC(=O)OO', 'OOC(=O)C1=CC(Cl)=CC=C1'], ['C([Cl])([Cl])']), 
    'Ox':       (oxid_Alkene, 14, [''], ['O=[Os](=O)(=O)=O', '[K+].[O-][Mn](=O)(=O)=O']), 
    'Oz_ox':       (ozon_ox_Alkene, 15, ['O=[O+][O-]'], ['OO']) , 
    'Oz_red':       (ozon_red_Alkene, 16, ['O=[O+][O-]'], ['CSC','NC(=S)N','[Pd].[H][H]','[Zn].O']) , # Need separate function to replace H with OH in advance for carboxylation 
    'H2':       (h2_Alkene_add, 10, ['[H][H]'], [ '[Pt]', '[Pd]','[Ni]','']),
    }

# Reaction dictionary:
# NR is a reaction that will be handled separately (return [Nd] as product)
Full_rxn_dict = { 
                  '1': sn2_rxn,
                  '2': E_rxn,
                  '3': sn_methyl_shift_rxn, 
                  '4': E_methyl_shift_rxn,
                }

for Mkwarg in Mark_rxn_dict.keys():
    Full_rxn_dict[ str(Mark_rxn_dict[Mkwarg][1])] = Mark_rxn_dict[Mkwarg][0]

for kwarg in other_rxn_dict.keys():
    Full_rxn_dict[ str(other_rxn_dict[kwarg][1])] = other_rxn_dict[kwarg][0]

