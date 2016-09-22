#!/bin/bash

#SBATCH -p aspuru-guzik
#SBATCH -n 1 
#SBATCH -t 48:00:00 
#SBATCH --mem=2000
#SBATCH -o neural1_balanced_200each_1.test
#SBATCH -e neural1_balanced_200each_1.err
#SBATCH -J neural1_bal_200each 

# To check:
#  - copy data files
#  - Data file percent matches train percent (and file names in hypopt file)
#  - fp type is correct and consistent in sub and hypopt

scfolder="/scratch/$(date +%Y%m%d%H%M)_neural_bal_200each_1/"
curr=$(pwd)
shoes=1

# Create scratch folder on cluster space
mkdir -p  $scfolder

#edit these
cp fp1_neural_hypopt_rxn_predict.py $scfolder 
cp fp1_reaction_estimator.py $scfolder

cp ~/reaction_learn/data/balanced_set/200each_class_3_2/balanced_200each_train_inputs_1.dat $scfolder/train_inputs.dat
cp ~/reaction_learn/data/balanced_set/200each_class_3_2/balanced_200each_train_targets.dat $scfolder/train_targets.dat

# execute python 
cd $scfolder

python -u fp1_neural_hypopt_rxn_predict.py > $curr/output/neural1_balanced_200each_train.out

# retrieve files back to local directory.
cd
mkdir -p  $curr/output/
cp  $scfolder/neural*  $curr/output/
rm -r $scfolder
