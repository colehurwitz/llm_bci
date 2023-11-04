import string
from seq_alignment import global_similarity

punctuation = string.punctuation.replace("'","")

def word_edit_distance(pred,target):

    pred = pred.translate(str.maketrans("","",punctuation)).lower().strip().split(" ")
    target = target.translate(str.maketrans("","",punctuation)).lower().strip().split(" ")

    sim_init = global_similarity(pred,target,False)
    sim_init.run()
    return sim_init.match_distance
