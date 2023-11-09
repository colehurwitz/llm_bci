import argparse
from tqdm import tqdm
from functools import partial

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from models.bci import BCI
from utils.data_utils import BCIDataset, pad_collate_fn
from utils.eval_utils import word_error_count

DEFAULT_CONFIG_FILE = "configs/default_finetune_config.yaml"

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():

    config = update_config(DEFAULT_CONFIG_FILE, args.config_file, "evaluate_config")

    savestring = "NDT2"
    epoch = 0

    # path_to_data = "/home/llm4bci/competitionData/processed.data"
    # path_to_model = "/home/llm4bci/LLM"
    # checkpoint_dir = "/home/llm4bci/checkpoints/NDT1/EP5"
    # ft_dir = "/home/llm4bci/ft_models/NDT1"

    path_to_data = "/n/home07/djimenezbeneto/lab/datasets/BCI/processed.data"
    path_to_model = "/n/home07/djimenezbeneto/lab/models/BCI"
    checkpoint_dir = f"/n/home07/djimenezbeneto/lab/BCI/checkpoints/{savestring}{epoch}"
    ft_dir = f"/n/home07/djimenezbeneto/lab/BCI/ft_models/{savestring}"
    
    seed = 1
    batch_size = 16
    max_new_tokens = 16
    eval_len = 16

    reset_seeds(seed)


    if epoch <= 0:
        # Load merged model
        model = BCI.from_pretrained(ft_dir)
    else:
        # Load from checkpoint
        model = BCI.from_pretrained(path_to_model, device_map="auto")
        model.load_adapter(checkpoint_dir)
        model.load_encoder(checkpoint_dir)

    model.to("cuda")
    model.eval()
    print(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path_to_model, padding_side='right')
    pad_id = tokenizer.eos_token_id

    # Load data
    data = torch.load(path_to_data)["test"]
    eval_dataset = BCIDataset(data, split="eval", len=eval_len)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=partial(pad_collate_fn,pad_id), batch_size=batch_size, pin_memory=True
    )

    # Compute word error rate
    all_preds = []
    all_sentences = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            model_inputs, sentences = batch
            model_inputs = {key: model_inputs[key].to("cuda") for key in model_inputs}
            preds = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False,pad_token_id=tokenizer.eos_token_id)
            preds = [tokenizer.decode(p.cpu().squeeze(), skip_special_tokens=True) for p in preds]
            all_preds += preds
            all_sentences += sentences
    

    torch.save({"preds": all_preds, "targets": all_sentences}, "preds.pt")
    for p, t in zip(all_preds[:10], all_sentences[:10]):
        print("Prediction: {}\nTarget: {}".format(p, t))

    errors, words = word_error_count(all_preds,all_sentences)
    print("Average error rate: {:.4f}".format(errors/words))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type = str, required=True, help="File (.yaml) with configurationfor finetuning")
    args = parser.parse_args()

    main(args)