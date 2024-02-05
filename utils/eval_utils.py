from copy import deepcopy

import torch
import torch.nn.functional as F

import editdistance


""" Compute word edit distance between source to target
"""
def word_edit_distance(source,target):
    source = source.split(" ")
    target = target.split(" ")
    return editdistance.eval(source, target), len(target)


""" Counts words and errors in a list of predicitons and targets
"""
def word_error_count(preds_0, targets_0):
    preds = deepcopy(preds_0)
    targets = deepcopy(targets_0)
    if not isinstance(preds, list):
        preds = [preds]
    if not isinstance(targets, list):
        targets = [targets]

    assert len(preds) == len(targets), "Lenghts of prediction and target lists don't match"
    errors = 0
    words  = 0
    for pred, target in zip(preds, targets):
        new_errors, new_words = word_edit_distance(pred.lower().strip(),target.lower().strip())
        errors += new_errors
        words += new_words

    # Return errors and words to allow for an accurate average over several calls
    return errors, words    
    
    
""" Convert prediciton of Neural Encoder to phonograms
"""
def format_ctc(pred, vocab, blank_id):
    phonogram = []
    last = -1
    for idx in pred:
        if idx != last and id != blank_id:
            phonogram.append(vocab[idx])
            last = deepcopy(idx)
    return phonogram

""" Get RMS between predicted rates and smoothed spiking data
"""
def smoothed_RMS(preds, features, targets_mask, width):
    
    # Create targets by averaging spike counts
    kernel =  torch.ones(width).view(1,1,-1)
    targets = []
    for f in features:
        targets.append((F.conv1d(f.unsqueeze(1), kernel, padding="same")/width).squeeze(1))
    targets = torch.stack(targets, 0 )


    mse = F.mse_loss(preds, targets, reduction="none")
    
    return (mse * targets_mask).sum().detach().item(), mse.sum().detach().item()
