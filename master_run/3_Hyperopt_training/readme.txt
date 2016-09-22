Scripts that were used to run hyperparameter optimization on Harvard's Odyssey Cluster, which uses SLURM to manage its jobs

# Running sklearn wrapper
fp1_double_hypopt_rxn_predict.py
fp1_double_neural_hypopt_rxn_predict.py
fp1_morgan_hypopt_rxn_predict.py
fp1_neural_hypopt_rxn_predict.py


# sklearn packet for the estimators (wrapper for the neural fingerprint code)
fp1_reaction_estimator.py
fp1_double_reaction_estimator.py

# Scripts for submitting to ODYSSEY
hypopt1_calc_neural_sub
hypopt1_calc_sub
hypopt2_calc_neural_sub
hypopt2_calc_sub
hypopt3_calc_double
hypopt3_calc_neural_double

# Viewing hyperopt results (using trials objects from hyperopt):
trials_view.py
