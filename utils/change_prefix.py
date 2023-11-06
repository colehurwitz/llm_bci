import torch
import re
import json

"""
Changes the prefix of the state dict of a llama chekpoint to match the names in the BCI architecture
"""

shards = [
    "pytorch_model-00001-of-00003.bin",
    "pytorch_model-00002-of-00003.bin",
    "pytorch_model-00003-of-00003.bin",
]

index_name = "pytorch_model.bin.index.json"


for shard_name in shards:
    shard = torch.load(shard_name)
    new_shard = {re.sub("model", "decoder.transformer", key): shard[key] for key in shard}
    new_shard_2 = {re.sub("lm_head", "decoder.lm_head", key): new_shard[key] for key in new_shard}
    torch.save(new_shard_2, shard_name)


with open(index_name, "r") as ifile:
    index = json.load(ifile)
    new_wmap = {re.sub("model", "decoder.transformer", key): index["weight_map"][key] for key in index["weight_map"]}
    new_wmap_2 = {re.sub("lm_head", "decoder.lm_head", key): new_wmap[key] for key in new_wmap}
    index["weight_map"] = new_wmap_2


with open(index_name, "w") as ofile:
    json.dump(index, ofile)

