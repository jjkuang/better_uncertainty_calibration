import os
import re
import pickle
# import torch


# Open file with pickled variables
def unpickle_probs(file, verbose=0):
    with open(file, 'rb') as f:
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)

    if verbose:
        print("y_preds_val:", y_probs_val.shape)  # (5000, 10); Validation set probabilities of predictions
        print("y_true_val:", y_val.shape)  # (5000, 1); Validation set true labels
        print("y_preds_test:", y_probs_test.shape)  # (10000, 10); Test set probabilities
        print("y_true_test:", y_test.shape)  # (10000, 1); Test set true labels

    return ((y_probs_val, y_val), (y_probs_test, y_test))


def logits_RC_to_pkl(logits_test, logits_test_RC, labels_test, file):
    with open(file, "wb") as ds_f:
        pickle.dump((logits_test, logits_test_RC, labels_test), ds_f)


# Open file with pickled variables
def unpickle_probs_RC(file, verbose=0):
    with open(file, 'rb') as f:
        (logits_test, logits_test_RC, y_true) = pickle.load(f)

    return (logits_test, logits_test_RC, y_true)


def logits_OOD_to_pkl(logits_test_iid, logits_test_ood, labels_test_iid, labels_test_ood, file):
    with open(file, "wb") as ds_f:
        pickle.dump((logits_test_iid, logits_test_ood, labels_test_iid, labels_test_ood), ds_f)


# Open file with pickled variables
def unpickle_probs_OOD(file, verbose=0):
    with open(file, 'rb') as f:
        (logits_test_iid, logits_test_ood, labels_test_iid, labels_test_ood) = pickle.load(f)

    return (logits_test_iid, logits_test_ood, labels_test_iid, labels_test_ood)


def parse_logits_path(logits_path):
    logits_settings = logits_path.split('_')
    model = logits_settings[-5] + '_' + logits_settings[-4]
    dataset = logits_settings[-3]
    precision = '_'.join(re.findall(r'\d+', logits_settings[-2]))
    qmethod = logits_settings[-1].rstrip('.p')

    return model, dataset, precision, qmethod


MODELS_MAP = {
                'vit_small_patch16_224': 'ViT-S',
                'swin_small_patch4_window7_224': 'Swin-S',
                'swin_tiny_patch4_window7_224': 'Swin-T',
                'deit_small_patch16_224': 'DeiT-S', 
                'deit_tiny_patch16_224': 'DeiT-T'
            }

METHODS_MAP = {
                'fqvit': 'FQ-ViT',
                'repqvit': 'RepQ-ViT',
                'iasvit': 'IaS-ViT'
            }

PRECISIONS_MAP = {
                    '32_32_32': 'FP32',
                    '8_8_32': 'w8a8att32',
                    '6_6_32': 'w6a6att32',
                    '8_4_32': 'w8a4att32',
                    '4_8_32': 'w4a8att32',
                    '4_4_32': 'w4a4att32',
                    '8_8_8': 'w8a8att8',
                    '6_6_6': 'w6a6att6',
                    '4_4_8': 'w4a4att8',
                    '8_4_8': 'w8a4att8',
                    '4_8_8': 'w4a8att8',
                    '8_4_4': 'w8a4att4',
                    '4_4_4': 'w4a4att4'
                }

