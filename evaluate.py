from tqdm import tqdm
from functools import partial

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from models.bci import BCI
from utils.data_utils import BCIDataset, pad_collate_fn
from utils.eval_utils import word_error_rate

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():

    path_to_model = "/n/home07/djimenezbeneto/lab/models/BCI"
    checkpoint_path = "/n/home07/djimenezbeneto/lab/BCI/checkpoints/NDT1/EP1"
    ft_path = "/n/home07/djimenezbeneto/lab/BCI/ft_models/NDT1
    
    
    proc_data_path = "/n/home07/djimenezbeneto/lab/datasets/BCI/processed.data"
    seed = 1
    split = "eval"
    batch_size = 16
    max_new_tokens = 16


    reset_seeds(seed)

    tokenizer = AutoTokenizer.from_pretrained(path_to_model, padding_side='right')

    # Load merged model
    # model = BCI.from_pretrained(path_to_ft)

    # Load from checkpoint
    model = BCI.from_pretrained(path_to_model)
    model.load_adapter(checkpoint_path)
    model.load_encoder(checkpoint_path)

    model.to("cuda")
    model.eval()
    print(model)

    # Arguments for padding function
    pad_id = tokenizer.eos_token_id

    # Eval DataLoader
    data = torch.load(proc_data_path)["test"]
    eval_dataset = BCIDataset(data, split="eval", len=64)
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=partial(pad_collate_fn,pad_id,split), batch_size=batch_size, pin_memory=True
    )


    # Compute word error rate
    all_preds = []
    all_sentences = []
    all_errors = 0
    all_words = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            model_inputs, sentences = batch
            model_inputs = {key: model_inputs[key].to("cuda") for key in model_inputs}
            output = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False,pad_token_id=tokenizer.eos_token_id)
            preds = [tokenizer.decode(output[i].cpu().squeeze(), skip_special_tokens=True) for i in range(len(output))]
            new_errors, new_words = word_error_rate(preds,sentences)
            all_errors += new_errors
            all_words += new_words
            all_preds += preds
            all_sentences += sentences
    
    for p, t in zip(all_preds[:10], all_sentences[:10]):
        print("Prediction: {}\nTarget: {}".format(p, t.replace(tokenizer.eos_token,"").replace(tokenizer.bos_token,"").strip()))

    print("Average error rate: {:.4f}".format(all_errors/all_words))
    


 
    # ## LOAD FROM ADAPTER
    # model = BCI.peft_from_adapter(model_name_or_path, adapter_path)
    # model.eval()
    # print(model.hf_device_map)
    # print(model)
    # output = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)[0]
    # print(tokenizer.decode(output.cpu().squeeze()))



if __name__ == "__main__":
    main()