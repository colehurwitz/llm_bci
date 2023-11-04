import string

import json

import numpy as np

vocab_path = "/home/gridsan/dbeneto/TFG/BCI/vocab125k.txt"
save_path = "/home/gridsan/dbeneto/MAML-Soljacic_shared/BCI/llm_bci/vocab125k_tree.json"
tokenizer_path = "/home/gridsan/dbeneto/MAML-Soljacic_shared/BCI/llama2-7b"

with open(vocab_path, "r") as vocab_file:
    lines = vocab_file.readlines()

words = sorted(set([line.split(" ")[0].translate(str.maketrans("","",string.punctuation)).lower().strip() for line in lines]))


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
word_ids  =  tokenizer(words, add_special_tokens=False)["input_ids"]
print(f"Different tokens usesd: {len(set([id for ids in word_ids for id in ids]))}")

# Recursive function to build tree
def add_word_to_tree(tree, word):
    # Finish condition
    if len(word) == 0:
        return
    
    # Create new leaf
    if word[0] not in tree.keys():
        tree[word[0]] = {}
    
    # Go down the tree
    add_word_to_tree(tree[word[0]], word[1:])
    
# Create tree
tree = {}
for word in word_ids:
    add_word_to_tree(tree, word)

json.dump(tree, open(save_path, "w"))

def random_word_from_tree(tree):
    word = []
    target = tree
    while len(word) == 0 or len(target.keys()) > 0:
        next_token = list(target.keys())[np.random.randint(len(target.keys()))]
        word.append(next_token)
        target = target[next_token]
    return word, tokenizer.decode(word)


