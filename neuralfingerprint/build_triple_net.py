'''
Goal: Building reaction fingerprints out of 3 neural fingerprints.

Method 1: Generate 3 fps: mol1, mol2, and reagents (with reagents added together)
    Implemented in this package

'''
from util import memoize, WeightsParser
from rdkit_utils import smiles_to_fps
from build_convnet import build_convnet_fingerprint_fun
from build_vanilla_net import build_fingerprint_deep_net

import autograd.numpy as np
import autograd.numpy.random as npr

def build_triple_morgan_fingerprint_fun(fp_length=512, fp_radius=4):

    def fingerprints_from_smiles(weights, smiles_tuple):
        # @pre : smiles_tuple is a list of three tuples 

        smiles1, smiles2, smiles3 = zip(*smiles_tuple)
        # Morgan fingerprints don't use weights.
        fp1 = fingerprints_from_smiles_tuple(tuple(smiles1))
        fp2 = fingerprints_from_smiles_tuple(tuple(smiles2))
        fp3 = fingerprints_from_smiles_tuple(tuple(smiles3))
        return np.concatenate([fp1, fp2, fp3], axis=1)

    @memoize # This wrapper function exists because tuples can be hashed, but arrays can't.
    def fingerprints_from_smiles_tuple(smiles_tuple):
        return smiles_to_fps(smiles_tuple, fp_length, fp_radius)

    return fingerprints_from_smiles


def build_triple_morgan_deep_net(fp_length, fp_depth, net_params):
    empty_parser = WeightsParser()
    morgan_fp_func = build_triple_morgan_fingerprint_fun(fp_length, fp_depth)
    return build_fingerprint_deep_net(net_params, morgan_fp_func, empty_parser, 0)


def build_triple_convnet_fingerprint_fun(**kwargs):

    fp_fun, parser = build_convnet_fingerprint_fun(**kwargs)

    def triple_fingerprint_fun(weights, smiles_tuple):
        smiles1, smiles2, smiles3 = zip(*smiles_tuple)
        fp1 = fp_fun(weights, smiles1)
        fp2 = fp_fun(weights, smiles2)
        fp3 = fp_fun(weights, smiles3)
        return np.concatenate([fp1, fp2], axis=1)

    return triple_fingerprint_fun, parser


def build_fixed_convnet_fingerprint_fun(**kwargs):

    fp_fun, parser = build_convnet_fingerprint_fun(**kwargs)

    random_weights = npr.RandomState(0).randn(len(parser))
    def triple_fingerprint_fun(empty_weights, smiles_tuple):
        smiles1, smiles2, smiles3 = zip(*smiles_tuple)
        fp1 = fp_fun(random_weights, smiles1)
        fp2 = fp_fun(random_weights, smiles2)
        fp3 = fp_fun(random_weights, smiles3)
        return np.concatenate([fp1, fp2, fp3], axis=1)

    empty_parser = WeightsParser()
    return triple_fingerprint_fun, empty_parser


def build_triple_conv_deep_net(conv_params, net_params, fp_l2_penalty=0.0):
    """Returns loss_fun(all_weights, smiles, targets), pred_fun, combined_parser."""
    #conv_fp_func, conv_parser = build_double_convnet_fingerprint_fun(**conv_params)
    conv_fp_func, conv_parser = build_fixed_convnet_fingerprint_fun(**conv_params)
    return build_fingerprint_deep_net(net_params, conv_fp_func, conv_parser, fp_l2_penalty)
