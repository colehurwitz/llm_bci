import os
from functools import partial
from tqdm import tqdm
from g2p_en import G2p

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM

from models.phoneme_llama import PhonemeLlama
from utils.config_utils import DictConfig, update_config
from utils.data_utils import PhonemesFinetuneDataset, ft_pad_collate_fn, prepare_phonemes_data
from utils.eval_utils import word_error_count


prompt = "phonemes: %% sentence:"
checkpoint_dir = "/n/home07/djimenezbeneto/lab/BCI/checkpoints"

savestring = "inter_1024-freeze_false-scale_2.0-lr_1.e-4-rank_1-drop_0.2-accum_4-size_1-stride_1"
EP = 6
STEP = 5004

checkpoint_dir = os.path.join(checkpoint_dir, savestring)
load_dir = os.path.join(checkpoint_dir,f"EP{EP}-STEP{STEP}")
config = DictConfig(torch.load(os.path.join(load_dir, "config.pth")))

config = update_config(config, "iaifi_dirs.yaml")
llama_dir = config.dirs.model_dir
tokenizer_dir = config.dirs.tokenizer_dir


config["trainer"]["test_len"] = -1

print("Model 1")
model = PhonemeLlama.from_pretrained(config.dirs.model_dir, config.coupler) 
model.load_coupler(load_dir)
adapter_file = os.path.join(load_dir, "adapter_config.json")
if os.path.isfile(adapter_file):
    model.load_adapter(load_dir, is_trainable=(not config.freeze_llm))


model.to("cuda")

# print("Model 2")
# llm = LlamaForCausalLM.from_pretrained(llm_dir)


tokenizer = AutoTokenizer.from_pretrained(config.dirs.tokenizer_dir, padding_side='right', add_bos_token=False, add_eos_token=False)
pad_id = tokenizer.eos_token_id
g2p = G2p()

data = torch.load(os.path.join(config.dirs.data_dir, config.data_file))
train_data = {k: v[:config.trainer.train_len] for k,v in data["train"].items()}
train_data = prepare_phonemes_data(train_data, tokenizer, g2p, prompt, config.stack)
test_data = {k: v[:config.trainer.test_len] for k,v in data["test"].items()}
test_data = prepare_phonemes_data(test_data, tokenizer, g2p, prompt, config.stack)

train_dataset = PhonemesFinetuneDataset(train_data)
test_dataset = PhonemesFinetuneDataset(test_data)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=partial(ft_pad_collate_fn, pad_id, "test"), batch_size=1, pin_memory=True,
)
test_dataloader = DataLoader(
    test_dataset, collate_fn=partial(ft_pad_collate_fn,pad_id,"test"), batch_size=1, pin_memory=True,
)

train_iter = iter(train_dataloader)
test_iter = iter(test_dataloader)


gen_config = {
    "do_sample": False, "max_new_tokens": 12,
    "temperature": 0.1, "repetition_penalty": 1.0, "length_penalty": -1.0, "renormalize_logits": True, "low_memory": True,
    "num_beams": 50, "num_beam_groups": 50, "diversity_penalty": 1.2,
    "no_repeat_ngram_size": None, 
    "num_return_sequences": 50,
    "output_scores": True, "return_dict_in_generate": True,
    "pad_token_id": pad_id
}

from time import perf_counter

all_pairs = []
all_sentences = []
all_errors = []
all_words = []
all_scores = []
# all_log_probs = []
for i, (model_inputs, prompt_inputs, sentence, true_ph, pred_ph) in tqdm(enumerate(test_dataloader)):
    a = perf_counter()
    prompt_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else [sub_v.to("cuda") for sub_v in v] for k,v in prompt_inputs.items()}
    preds = model.generate(**prompt_inputs, **gen_config)
    prompt_lens = [len(prompt_inputs["input_ids"][i]) for i in range(len(prompt_inputs["input_ids"]))]*gen_config["num_return_sequences"]
    dec_preds = [tokenizer.decode(p[prompt_lens[i]:].detach().cpu().squeeze(), skip_special_tokens=True) for i, p in enumerate(preds.sequences)]
    # log_probs = []
    # b = perf_counter()
    # for p in dec_preds:
    #     log_prob = 0.
    #     t = tokenizer(p, return_tensors="pt")
    #     out = llm(**t)
    #     prob = torch.nn.functional.log_softmax(out.logits, dim=2)
    #     for i, token in enumerate(t["input_ids"][0]):
    #         log_prob += prob[0,i,t["input_ids"][0,i]].item()
    #     log_probs.append(log_prob)
    scores = preds.sequences_scores
    scores = (scores-scores.mean())/scores.std()
    pairs = sorted([(a.item(), b) for a,b in zip(scores, dec_preds)], key=lambda x: -x[0])
    # c = perf_counter()
    # print("Times: ", b-a, c-b)
    print(sentence)
    for pair in pairs:
        print(pair[0], pair[1])
    errors, words = word_error_count(pairs[0][1], sentence)
    all_errors.append(errors)
    all_words.append(words)
    all_pairs.append(pairs)
    all_sentences.append(sentence)
    all_scores.append(scores.tolist())
    # all_log_probs.append(log_probs)
    torch.save({"errors": all_errors, "words": all_words, "pairs": all_pairs, "sentences": [s for [s] in all_sentences], "scores": all_scores}, "data.pth")


llama_errors = 0
llama_words = 0
for s, p in zip(sentences, pairs):
    best = 100
    best_pred = None
    for ex in p:   
        errors, words = word_error_count(ex[1].strip(), s.strip())    
        if errors < best:
            best = errors
            best_pred = ex[1]
    llama_errors += best
    llama_words += words
    print(s, best_pred, best)

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



llama_errors = 0
llama_words = 0
for s, p in zip(sentences, pairs):  
    sorted_p = sorted(p, key=lambda x: -x[0])
    errors, words = word_error_count(sorted_p[0][1].strip(), s.strip())    
    llama_errors += errors
    llama_words += words
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


vocab = [
    'AA', 'AE', 'AH', 'AO', 'AW', 
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K', 
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH', 
    'T', 'TH', 'UH', 'UW', 'V', 
    'W', 'Y', 'Z', 'ZH', 'SIL', 'BLANK'
]
all = []
from scipy.special import softmax
import numpy as np
split="train"
for i in range(len(d[split]["phoneme_logits"])):
s = softmax(d[split]["phoneme_logits"][i], axis=-1)
[vocab[a] for a in np.argmax(s,axis=-1)]
np.sort(s[6])
first = np.argmax(s,-1)
second = np.argsort(s)[:,-2]
    all.append(np.mean(s[np.arange(len(s)),first]/s[np.arange(len(s)),second]))

sum(all)/len(all)