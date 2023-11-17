from copy import deepcopy

from seq_alignment import global_similarity


""" Compute word edit distancefrom source to target
"""
def word_edit_distance(source,target):

    source = source.split(" ")
    target = target.split(" ")

    sim_init = global_similarity(source,target,False)
    sim_init.run()
    return sim_init.match_distance, len(target)


""" Counts words and errors in a list of predicitons and targets
"""
def word_error_count(preds, targets):

    if not isinstance(preds, list):
        preds = [preds]
    if not isinstance(targets, list):
        targets = [targets]

    assert len(preds) == len(targets), "Lenghts of prediction and target lists don't match"
    errors = 0
    words  = 0
    for pred, target in zip(preds, targets):
        new_errors, new_words = word_edit_distance(pred,target)
        errors += abs(new_errors)
        words += new_words

    # Return errors and words to allow for an accurate average over several calls
    return errors, words    
    
    
""" Convert prediciton of Neural Encoder to phonograms
"""
def format_ctc(pred, vocab):
    phonogram = []
    last = -1
    for i in pred:
        if i != last:
            phonogram.append(vocab[i])
            last = deepcopy(i)
    return phonogram