import torch
import re
import json


shards = [
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin"
]

index_name = "pytorch_model.bin.index.json"


for shard_name in shards:
    shard = torch.load(shard_name)
    new_shard = {re.sub("model", "decoder", key): shard[key] for key in shard}
    torch.save(new_shard, shard_name)


with open(index_name, "r") as ifile:
    index = json.load(ifile)
    new_wmap = {re.sub("model", "decoder", key): index["weight_map"][key] for key in index["weight_map"]}
    index["weight_map"] = new_wmap


with open(index_name, "w") as ofile:
    json.dump(index, ofile)

