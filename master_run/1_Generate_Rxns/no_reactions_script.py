'''

Valid reactions will be handled separately by another script.

# Compiling all reactions that should be null.

1. Compile all possible reactants and reagents in separate lists
2. Compile all 'valid' combinations as list of tuples:
        ( alk type, reactant, reagent)
3. Loop over all combos of reactants and reagents to write null reactions
        - Skip reactions that are part of valid set.
        - Also need to loop over molecule type.

'''

from alkene_rxn_components import Mark_rxn_dict, other_rxn_dict  


# combining all reactants
AllAlkeneRxns_dict = dict(Mark_rxn_dict, **other_rxn_dict)
All_alkene_rcts = []
All_alkene_rgts = []

# Create pairs of valid reactions for alkenes. 
alkene_valid_reactions = []

for kwarg in AllAlkeneRxns_dict: 
    All_alkene_rcts.extend(AllAlkeneRxns_dict[kwarg][2])
    All_alkene_rgts.extend(AllAlkeneRxns_dict[kwarg][3])

    for rct in AllAlkeneRxns_dict[kwarg][2]:
        for rgt in AllAlkeneRxns_dict[kwarg][3]:
            alkene_valid_reactions.append((rct, rgt))

all_reactant_list = list(set(All_alkene_rcts))
all_reagent_list = list(set(All_alkene_rgts))

#print all_reactant_list
#print all_reagent_list

#print 'alkene reaction sets'
#print alkene_valid_reactions

alkylhalide_reactant_list = [ '[C-]#N', '[O-]', 'O', 'C(C)(C)C[O-]']
alkylhalide_reagent_list = ['']

all_reactant_list.extend(alkylhalide_reactant_list)
all_reagent_list.extend(alkylhalide_reagent_list)

alkylhalide_valid_reactions = [(rct, alkylhalide_reagent_list[0]) for rct in alkylhalide_reactant_list]
    
swap = False 
display_prods = True
show_rxn_vec = True 
num_rxn_types = 18 

#print 'alkylhalide reaction sets'
#print alkylhalide_valid_reactions 

# Creating all possible combos of reactants and reagents:

alkylhal_rxn_tups = []
alkene_rxn_tups = []

# Making list of reactant/reagent pairs
for rct in all_reactant_list:
    for rgt in all_reagent_list:
        new_rct_rgt_tup = (rct, rgt)
        if (new_rct_rgt_tup in alkylhalide_valid_reactions) and ( new_rct_rgt_tup not in alkene_valid_reactions):
            alkene_rxn_tups.append(new_rct_rgt_tup)
        elif (new_rct_rgt_tup not in alkylhalide_valid_reactions) and ( new_rct_rgt_tup in alkene_valid_reactions):
            alkylhal_rxn_tups.append(new_rct_rgt_tup)
        elif (new_rct_rgt_tup not in alkylhalide_valid_reactions) and ( new_rct_rgt_tup not in alkene_valid_reactions):
            alkene_rxn_tups.append(new_rct_rgt_tup)
            alkylhal_rxn_tups.append(new_rct_rgt_tup)


def write_rxn_NR(mol_smi, other_rct_smi, reagent_smi):
    #swap = True

    rxn_vec = '1.0'
    for it in range(num_rxn_types-1):
        rxn_vec += ',0.0'

    if mol_smi == '': mol_smi = '[Nd]'
    if other_rct_smi == '': other_rct_smi = '[Nd]'

    #!# Establish COC as default solvent
    if reagent_smi == '': reagent_smi = 'COC'
    else: reagent_smi += '.COC'
    
    rxn_smi = mol_smi + '.' + other_rct_smi + '>' + reagent_smi + '>'

    if display_prods:
        rxn_smi += '[Nd]' 
    if show_rxn_vec:
        rxn_smi += ','+rxn_vec

    if swap:
        rxn_smi += '\n' + other_rct_smi + '.' + mol_smi + '>' + reagent_smi + '>' 
        if display_prods:
            rxn_smi += '[Nd]' 
        if show_rxn_vec:
            rxn_smi += ','+rxn_vec

    return rxn_smi


#NR_reaction_f = open('all_rxns/NR_rxns_noswap_vec.dat','a')
#NR_reaction_f = open('03_22_rxns/NR_rxns_swap_vec.dat','a')
NR_reaction_f = open('balanced_set/NR_rxns_noswap_vec.dat','w')
#NR_reaction_f = open('example_NRs.dat','a')


target_name_array = ['prob_'+str(i) for i in range(num_rxn_types)]
header = 'smiles,'+','.join(target_name_array) + '\n'
NR_reaction_f.write(header)

# Writing NR reactions:
with open('alkene_0.txt') as f:
    alkenes = f.read().splitlines()

with open('alkylhalides.dat') as f2:
    alkylhals = f2.read().splitlines() 


for alkene in alkenes:
    for rxn_tup in alkene_rxn_tups:
        NR_reaction_f.write(write_rxn_NR(alkene, rxn_tup[0], rxn_tup[1]) + '\n') 


for alkylhal in alkylhals:
    for rxn_tup in alkylhal_rxn_tups:
        NR_reaction_f.write(write_rxn_NR(alkylhal, rxn_tup[0], rxn_tup[1]) + '\n') 
        
NR_reaction_f.close()
