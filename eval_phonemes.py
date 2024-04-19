import os
from functools import partial
from tqdm import tqdm
from g2p_en import G2p

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.phoneme_llm import PhonemeLLM
from utils.config_utils import DictConfig, update_config
from utils.data_utils import PhonemesFinetuneDataset, ft_pad_collate_fn, prepare_phonemes_data
from utils.eval_utils import word_error_count

# torch.backends.cudnn.benchmark = True

prompt = "phonemes: %% sentence:"
checkpoint_dir = "/n/home07/djimenezbeneto/lab/BCI/checkpoints"
checkpoint_dir = "/home/gridsan/dbeneto/TFG/BCI/checkpoints"

savestring = "new_ft/eval_64-accum_4-rank_1-lr_1.e-4-gauss_0.0-spikes_0.6_2_0.8-norm_identity_1"
STEP = 36249

checkpoint_dir = os.path.join(checkpoint_dir, savestring)
load_dir = os.path.join(checkpoint_dir,f"STEP{STEP}")
config = DictConfig(torch.load(os.path.join(load_dir, "config.pth")))

# config = update_config(config, "iaifi_dirs.yaml")
# config = update_config(config, "configs/sc_dirs.yaml")
llama_dir = config.dirs.llm_dir
tokenizer_dir = config.dirs.tokenizer_dir

config["dirs"]["llm_dir"] = "/home/gridsan/dbeneto/MAML-Soljacic_shared/llama2/7b"
llm = AutoModelForCausalLM.from_pretrained(config.dirs.llm_dir)
model = PhonemeLLM(llm, load_dir)
adapter_file = os.path.join(load_dir, "adapter_config.json")
if os.path.isfile(adapter_file):
    model.load_lora_adapter(load_dir, is_trainable=False)

    
model.to("cuda")


tokenizer = AutoTokenizer.from_pretrained(config.dirs.tokenizer_dir, add_bos_token=False, add_eos_token=False)
pad_id = tokenizer.eos_token_id
g2p = G2p()


config["trainer"]["test_len"] = -1
data = torch.load(os.path.join(config.dirs.data_dir, config.data_file))
train_data = {k: v[:config.trainer.train_len] if config.trainer.train_len != -1 else v for k,v in data["train"].items()}
train_data = prepare_phonemes_data(train_data, tokenizer, g2p, config.prompt)
test_data = {k: v[:config.trainer.test_len] if config.trainer.test_len != -1 else v for k,v in data["test"].items()}
test_data = prepare_phonemes_data(test_data, tokenizer, g2p, config.prompt)
train_dataset = PhonemesFinetuneDataset(train_data)
test_dataset = PhonemesFinetuneDataset(test_data)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=partial(ft_pad_collate_fn,config.noise,config.mask,pad_id,"test"), batch_size=1, pin_memory=True,
)
test_dataloader = DataLoader(
    test_dataset, collate_fn=partial(ft_pad_collate_fn,config.noise,config.mask,pad_id,"test"), batch_size=1, pin_memory=True,
)

train_iter = iter(train_dataloader)
test_iter = iter(test_dataloader)


beams = 5
gen_config = {
    "max_new_tokens": 20, 
    "do_sample": False, #"temperature": 1.0,  "top_p": 0.6, "top_k": 40, 
    "num_beams": beams, 
    "num_beam_groups": beams, "diversity_penalty": 1.2,
    "repetition_penalty": 1.0, "length_penalty": 1.0, "no_repeat_ngram_size": 2, 
    "renormalize_logits": True, 
    "low_memory": True,
    "num_return_sequences": beams, "output_scores": True, "return_dict_in_generate": True,
    "pad_token_id": pad_id
}

from time import perf_counter

all_pairs = []
all_sentences = []
all_errors = []
all_words = []
all_scores = []
time_b = 0.
time_c = 0.
for i, (model_inputs, prompt_inputs, sentence, true_ph, pred_ph) in tqdm(enumerate(test_dataloader)):
    a = perf_counter()
    prompt_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else [sub_v.to("cuda") for sub_v in v] for k,v in prompt_inputs.items()}
    preds = model.generate(**prompt_inputs, **gen_config, synced_gpus=None)
    b = perf_counter()
    time_b += b-a 
    dec_preds = [tokenizer.decode(p.detach().cpu().squeeze(), skip_special_tokens=True) for i, p in enumerate(preds.sequences)]
    # print(dec_preds)
    scores = preds.sequences_scores if "sequences_scores" in dir(preds) else torch.zeros(len(preds))
    scores = (scores-scores.mean())/scores.std()
    pairs = sorted([(a.item(), b) for a,b in zip(scores, dec_preds)], key=lambda x: -x[0])
    print(sentence)
    for pair in pairs:
        print(pair[0], pair[1])
    errors, words = word_error_count(pairs[0][1], sentence)
    all_errors.append(errors)
    all_words.append(words)
    all_pairs.append(pairs)
    all_sentences += sentence
    all_scores.append(scores.tolist())
    c = perf_counter()
    time_c += c-b


print(time_b, time_c)
torch.save({"errors": all_errors, "words": all_words, "pairs": all_pairs, "sentences": all_sentences, "scores": all_scores}, f"data{beams}.pth")


# model.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_config, synced_gpus=synced_gpus)

llm.to("cuda")
inputs = tokenizer(tokenizer.bos_token + "My name is", return_tensors="pt")
inputs = {k: v.to("cuda") for k,v in inputs.items()}
# tokenizer.batch_decode(llm.generate(**inputs, max_new_tokens=10))
inputs_embeds = llm.get_input_embeddings()(inputs["input_ids"])
tokenizer.batch_decode(llm.generate(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"], max_new_tokens=20), skip_special_tokens=True)


import torch
from utils.eval_utils import word_error_count
d = torch.load("data5.pth")
all_sentences = d["sentences"]
all_pairs = d["pairs"]
llama_errors = 0
llama_words = 0
a=0
b=-1
for s, p in zip(all_sentences[a:b], all_pairs[a:b]):
    best = 100
    best_pred = None
    for ex in p:   
        errors, words = word_error_count(ex[1].strip(), s.strip())    
        if errors < best:
            best = errors
            best_pred = ex[1]
    llama_errors += best
    llama_words += words
    print(s," // ", best_pred, best)


llama_errors/llama_words

# ngram_errors = 0
# ngram_words = 0
# for s, p in tqdm(zip(sentences, ngram_pairs)):
#     best = 100
#     best_pred = None
#     for ex in p:   
#         errors, words = word_error_count(ex[0].strip(), s.strip())    
#         if errors < best:
#             best = errors
#             best_pred = ex[0]
#     ngram_errors += best
#     ngram_words += words
#     print(f"{s}\n {best_pred}\n  {best}")

# ngram_errors/ngram_words

words = 0
errors = 0

for p,s in zip(preds, sentences):
    e, w = word_error_count(p,s)
    words += w
    errors += e


errors/words

# new_log_probs = []
# for l in log_probs:
#     log = torch.tensor(l)
#     new_log_probs.append(((log - log.mean())/log.std()).tolist())

# new_pairs = []
# for i, p in enumerate(pairs):
#     new_p = []
#     for ex, l in zip(p, new_log_probs[i]):
#         new_p.append((ex[1], ex[0], l))
#     new_pairs.append(new_p)



llama_errors = []
llama_words = []
wer =[]
for s, p in zip(all_sentences, all_pairs):  
    sorted_p = sorted(p, key=lambda x: -x[0])
    errors, words = word_error_count(sorted_p[0][1].strip(), s.strip())    
    llama_errors.append(errors)
    llama_words.append(words)
    wer.append(errors/words)
    # print(s, sorted_p[0][0], errors)


llama_errors/llama_words



# alfa_errors = []
# for alfa in [0,1]:
#     ngram_errors = 0
#     ngram_words = 0
#     for s, p in tqdm(zip(sentences, ngram_pairs)):  
#         sorted_p = sorted(p, key=lambda x: -x[1]*alfa - x[2])
#         errors, words = word_error_count(sorted_p[0][0].strip(), s.strip())    
#         ngram_errors += errors
#         ngram_words += words
#         # print(s, sorted_p[0][0], errors)
#     ngram_errors/ngram_words
#     alfa_errors.append((alfa,ngram_errors/ngram_words))

# alfa_errors


# vocab = [
#     'AA', 'AE', 'AH', 'AO', 'AW', 
#     'AY', 'B', 'CH', 'D', 'DH',
#     'EH', 'ER', 'EY', 'F', 'G',
#     'HH', 'IH', 'IY', 'JH', 'K', 
#     'L', 'M', 'N', 'NG', 'OW',
#     'OY', 'P', 'R', 'S', 'SH', 
#     'T', 'TH', 'UH', 'UW', 'V', 
#     'W', 'Y', 'Z', 'ZH', 'SIL', 'BLANK'
# ]
# all = []
# from scipy.special import softmax
# import numpy as np
# split="train"
# for i in range(len(d[split]["phoneme_logits"])):
# s = softmax(d[split]["phoneme_logits"][i], axis=-1)
# [vocab[a] for a in np.argmax(s,axis=-1)]
# np.sort(s[6])
# first = np.argmax(s,-1)
# second = np.argsort(s)[:,-2]
#     all.append(np.mean(s[np.arange(len(s)),first]/s[np.arange(len(s)),second]))

# sum(all)/len(all)



lenghts = {}
for s in sentences:
    l = len(tokenizer(s)["input_ids"])
    if l in lenghts:
        lenghts[l] += 1
    else:
        lenghts[l] = 1
