'''
Pseudocode:

Have list of questions
1. Load question answer
        Question answers in one txt file, each it's own row
        Parse out and get list --> Convert to vector
2. Load predicitons
    Standardized naming format:
        {batch name}_{prediction method}_{question}.dat
        e.g. class_3_1_bl_rct2_Wade8_47.dat
3. Measure accuracy of predictions (what are scores for each answer?)
    - For multiple answer, may be better to just normalize 
4. Return array (problems as rows, methods as columns)
'''
import pickle as pkl
import ast
import numpy as np
from scipy.misc import logsumexp
from neuralfingerprint.parse_data import confusion_matrix

def parse_answer_from_line(answer_line):
    # Once line has been identified, create the answer key
    answer_list_str = answer_line.split('=')[1].strip()
    answer_list = ast.literal_eval(answer_list_str)
    return answer_list
   

def Get_answer_from_name(prob_str, ans_lines):
    # match the problem string in the lines to parse the answer
    # @inp ans_lines: All answers from the input

    for ii in range(len(ans_lines)):
        if prob_str in ans_lines[ii]: 
            return parse_answer_from_line(ans_lines[ii])
    
    else:
        print 'error, no such answer key'
        return []


def one_of_K_encoding(answers, num_outputs):
    # Answers provided in list of single value or tuple
    rxn_set = [i for i in range(num_outputs)] 
    rxn_vec = np.zeros(num_outputs)

    if type(answers) == tuple:
        # handle multiple tuple
        for ii in range(len(answers)):
            # If we just want to record all correct answers for matching, should just add
            rxn_vec[answers[ii]] = 1. #/(len(answers))
    else:
        rxn_vec[answers] = 1.
    return rxn_vec


def makeAnsMatrix(answers_list, num_outputs):
    # input answers as a list of answers, entries are either ints or tups
    
    ans_matrix = np.zeros((len(answers_list),num_outputs))

    for prob_subid in range(len(answers_list)):
        ans_matrix[prob_subid] = one_of_K_encoding(answers_list[prob_subid], num_outputs)

    return ans_matrix


def Generate_pred_fnames(problem_name, batch_header, methods_list):
    return [batch_header+'_'+method_name+'_'+problem_name+'.dat' for method_name in method_list]


def get_normalized_pred(unnorm_pred_mat):
    # Normalizes the prediction matrix (which is currently reported as log of probability and unormalized
    pred_mat = np.exp(unnorm_pred_mat  - logsumexp(unnorm_pred_mat, axis=1, keepdims=True))
    return pred_mat


def calculate_accuracy(question_pred, question_ans):
    # @brief : Calculate the accuracy of prediction of expected reaction 
    # @inp question_pred : matrix of predictions of each rxn type for each question
    # @inp question_ans : matrix of answers for each rxn type for each question
    # @out : A vector of accuracy calculations

    # convert predictions to not exponent form
    question_pred_normed = get_normalized_pred(question_pred) 
    print question_pred_normed
    print question_ans
    
    return np.sum(question_pred_normed*question_ans, axis=1) 


if __name__ == '__main__':
    num_rxns = 18

    batch_header = '200_each'
    method_list =  ['morgan', 'neural'] 
                    #'bl_rgt', 'bl_rct2', 'bl_rgt_rct2'] 

    # Load answers
    problem_list = ['Wade8_47', 'Wade8_48']
    with open('answers.txt') as ansf:
       answers = ansf.readlines() 

    # Within each problem
    for problem in problem_list:
        ans_list = Get_answer_from_name(problem, answers)
        this_ans_matrix = makeAnsMatrix(ans_list, num_rxns) 

        # Loading all associated predictions
        # Generating pred_fs name:
        pred_fnames = Generate_pred_fnames(problem, batch_header, method_list)
        print pred_fnames

        this_pred_mats = [pkl.load(open('results/'+pred_fnames[f_idx])) for f_idx in range(len(pred_fnames))]
       
        ans_acc_mat = np.zeros((len(ans_list), len(method_list)))
        for pred_idx in range(len(method_list)):
            # Load method prediction
            # Calculate the predction of the accuracy
            ans_acc_mat[:, pred_idx] = calculate_accuracy(this_pred_mats[pred_idx], this_ans_matrix)

        # Save the matrix:
        with open(batch_header+'_'+problem+'.dat','w') as this_prob_ansf:
            pkl.dump(ans_acc_mat, this_prob_ansf)
    

