Scripts that were used to run hyperparameter optimization on Harvard's Odyssey Cluster, which uses SLURM to manage its jobs. 

# Scripts for running sklearn estimator
fp1_double_hypopt_rxn_predict.py   - Use these if you want to use  two molecular Morgan fingerprint inputs
fp1_double_neural_hypopt_rxn_predict.py  - Use these if you want to use two molecular neural fingerprint inputs
fp1_morgan_hypopt_rxn_predict.py   - Use these if you want to use 
fp1_neural_hypopt_rxn_predict.py

# sklearn wrapper of Regressor object for reaction estimation (wrapper for the neural fingerprint code) 
#  required by all the codes above
fp1_reaction_estimator.py
fp1_double_reaction_estimator.py

# Scripts for submitting to Harvard ODYSSEY supercomputer. If you use these, be sure to change the directories for the files/data files accordingly
hypopt1_calc_neural_sub - Runs fp1_neural_hypopt_rxn_predict.py and corresponding files
hypopt1_calc_sub        - Runs fp1_morgan_hypopt_rxn_predict.py and corresponding files
hypopt2_calc_neural_sub - A duplicate of hypopt1_calc_neural_sub, but will run on second dataset that takes reactants and products as inputs. All products are considered one molecule
hypopt2_calc_sub        - A duplicate of hypopt1_cal_sub, but will run on second dataset that takes reactants and products as inputs.
hypopt3_calc_double     - Runs fp1_double_hypopt_rxn_predict.py; Use if you want to use just two molecular fingerprint inputs.
hypopt3_calc_neural_double  - Runs fp1_double_neural_hypopt_rxn_predict.py; Use if you want to use just two molecular fingerprint inputs.

# Viewing hyperopt results (using trials objects from hyperopt):
trials_view.py          - Run at the end of hyperparameter optimization to see best results
