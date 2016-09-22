'''
3/28 update, change so matches the textbook
    - Add more reactions (like with the reduction complex)
    - Add only reagents explicitly mentioned in the textbook
'''


from rdkit import Chem
from rdkit.Chem import AllChem
from toolkit import MarkMarkovAtom, GetDoubleBondAtoms, get_molecule_smi


####################
#### Alkene reactions:
#    Should come up with a library for alcohols

## Markovnikov/ AntiMarkovnikov sensitive
# To add: M - Markovnikov;   AM - anti-Markovnikov
# Hydrohalogenation (hyd_hal)
# Requires [Si] marker
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
    'halhal':   (halhal_Alkene, 11, ['BrBr','ClCl', 'II'], ['','C([Cl])([Cl])']),
    'epox':     (epox_Alkene, 13,  ['CC(=O)OO', 'CCC(=O)OO', 'CCCC(=O)OO', 'OOC(=O)C1=CC(Cl)=CC=C1'], ['C([Cl])([Cl])']), 
    'Ox':       (oxid_Alkene, 14, [''], ['O=[Os](=O)(=O)=O', '[K+].[O-][Mn](=O)(=O)=O']), 
    'Oz_ox':       (ozon_ox_Alkene, 15, ['O=[O+][O-]'], ['OO']) , 
    'Oz_red':       (ozon_red_Alkene, 16, ['O=[O+][O-]'], ['CSC','NC(=S)N','[Pd].[H][H]','[Zn].O']) , # Need separate function to replace H with OH in advance for carboxylation 
    'H2':       (h2_Alkene_add, 10, ['[H][H]'], [ '[Pt]', '[Pd]','[Ni]','']),
    }

# combine all the other reactants that are not part of the main molecule of interest.

AllAlkeneRxns_dict = dict(Mark_rxn_dict, **other_rxn_dict)
All_alkene_rcts = []
All_alkene_rgts = []

for kwarg in AllAlkeneRxns_dict: 
    All_alkene_rcts.extend(AllAlkeneRxns_dict[kwarg][2])
    All_alkene_rgts.extend(AllAlkeneRxns_dict[kwarg][3])

alkene_reactant_list = list(set(All_alkene_rcts))
alkene_reagent_list = list(set(All_alkene_rgts))

alkylhalide_reactant_list = [ '[C-]#N', '[O-]', 'O', '[O-]C(C)(C)C']

