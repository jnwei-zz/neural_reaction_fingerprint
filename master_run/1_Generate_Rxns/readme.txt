6/5/16:
update 11/3/16: Added generate_substrates.py so new substrates may be generated

Compiling code used to generate reaction datasets for each reaction. 

Main code:
separate_alkene_rxns.py   - Generate all alkene reactions as separate files
separate_alkylhal_rxns.py - Generate all alkylhalide reactions as separate files
no_reactions_script.py    - Generate all NR reactions (across different reaction sets and alkylhalide/alkene)

Toolkits:
alkene_rxn_components.py  - Holds all alkene reactions and reaction sets (corresponding alkylhalide is inside of separate_alkylhalide and no_reactions 
toolkit.py                - Holds tools for identifying Markov products 
generate_substrates.py    - Generate substrate files from scratch

Substrate files:
alkene_0.txt
alkylhalides.dat
