import string
from seq_alignment import global_similarity

punctuation = string.punctuation.replace("'","")

# Compute word edit distance bewtween two sentences afet normalising to the dataset 
# (lowercase and remove punctuation except apostrophes)
def word_edit_distance(pred,target):

    pred = pred.translate(str.maketrans("","",punctuation)).lower().strip().split(" ")
    target = target.translate(str.maketrans("","",punctuation)).lower().strip().split(" ")

    sim_init = global_similarity(pred,target,False)
    sim_init.run()
    return sim_init.match_distance, len(target)


# Compute word error rate over a list of predicitons and targets
def word_error_rate(preds, targets):

    assert len(preds) == len(targets), "Lenghts of prediction and target lists don't match"
    errors = 0
    words  = 0
    for pred, target in zip(preds, targets):
        new_errors, new_words = word_edit_distance(pred,target)
        errors += abs(new_errors)
        words += new_words

    return errors, words
    
    