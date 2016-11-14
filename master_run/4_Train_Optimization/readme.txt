
baseline_2.py :
    - Code for running the baseline test using the one-hot representation of reagents/reactants.

linear_regression_estimator.py:
    - library required for baseline_2.py (linear regression estimator in form of scikit learn estimator)

single_run.py : 
    - Code for production run once hyperparameters have been selected
    - requires fp1_reaction_estimator in neuralfingerprint ; this file is a copy of the same file in the 3_Hyperopt_training directory

fancy_cmat_gen.py :
    - Code for generating a confusion matrix of the cross-validation data
