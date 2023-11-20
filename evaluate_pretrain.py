import os
import argparse
from tqdm import tqdm
from functools import partial

import torch
from torch.utils.data import DataLoader

from models.neural_encoder import NeuralEncoder
from models.neural_pretrainer import NeuralPretrainer

from utils.config_utils import update_config, config_from_kwargs, ParseKwargs, DictConfig
from utils.eval_utils import format_ctc, word_error_count, smoothed_RMS
from utils.data_utils import NeuralPretrainerDataset, pt_pad_collate_fn


DEFAULT_CONFIG_FILE = "configs/default_evaluate_pretrain_config.yaml"

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args):

    # Evalaution configs
    config = update_config(DEFAULT_CONFIG_FILE, args.config_file if args.config_file != "none" else None)
    config = update_config(config, config_from_kwargs(args.kwargs))

    reset_seeds(config.seed)

    # Checkpoint dirs
    if config.checkpoint_epoch == None:
        model_dir = os.path.join(config.pt_dir,config.savestring)
    else:
        model_dir = os.path.join(config.checkpoint_dir,config.savestring,"EP"+str(config.checkpoint_epoch))

    # Configs used in pretraining
    pretrain_config = DictConfig(torch.load(os.path.join(model_dir,"config")))
    config.neural_config = update_config(config.neural_config, pretrain_config.neural_config)
    config.neural_pretrainer = update_config(config.neural_pretrainer, pretrain_config.neural_pretrainer)
    print(config)

    # Load data
    eval_data = torch.load(os.path.join(config.data_dir, config.data_file))[config.split]
    
    vocab = eval_data["eval"]["vocab"]
    blank_id = vocab.index("BLANK") if config.neural_pretrainer.loss.type == "ctc" else None
    vocab_size = len(vocab)  if config.neural_pretrainer.loss.type == "ctc" else None

    eval_dataset = NeuralPretrainerDataset(eval_data, loss_fn=config.neural_pretrainer.loss.type, len=config.eval_len)

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=partial(pt_pad_collate_fn,blank_id), batch_size=config.eval_batch_size, pin_memory=True
    )   


    encoder = NeuralEncoder(config.neural_config)
    model = NeuralPretrainer(encoder, config.neural_pretrainer, vocab_size, blank_id)
    print(f"Loading model from {model_dir}")
    model.encoder.load_state_dict(torch.load(os.path.join(model_dir,"encoder.bin")))
    model.decoder.load_state_dict(torch.load(os.path.join(model_dir,"decoder.bin")))

    eval_loss = 0.
    eval_RMS = 0.
    eval_mask_RMS = 0.
    eval_examples = 0
    eval_mask_examples = 0
    eval_errors = 0.
    eval_phonemes = 0

    for step, (batch, phonograms, sentences) in enumerate(tqdm(eval_dataloader)):

        # Forward pass
        with torch.no_grad():
            outputs = model(**batch)
        
        # Gather test metrics
        eval_loss += outputs.loss.detach().float()
        eval_examples += outputs.n_examples

        if config.neural_pretrainer.loss.type == "ctc":
            preds = torch.argmax(outputs.outputs,-1)
            preds = [" ".join(format_ctc(pred, vocab, blank_id)) for pred in preds]
            phonograms = [" ".join(p) for p in phonograms]
            errors, phonemes = word_error_count(preds, phonograms)
            eval_errors += errors
            eval_phonemes += phonemes

            # for p, s in zip(preds, sentences):
            #     print(f"Sentence: {s} \nPrediction: {p} \n")
        elif config.neural_pretrainer.loss.type == "poisson":
            targets_mask = outputs.targets_mask
            features = batch["features"]
            preds = outputs.outputs
            mask_RMS, RMS = smoothed_RMS(preds, features, targets_mask, config.smoothing, config.neural_pretrainer.use_lograte)
            eval_RMS += RMS
            eval_mask_RMS += mask_RMS
            eval_examples += features.nelement()
            eval_mask_examples += targets_mask.sum()
        
        

    # Log to tensorboard
    eval_epoch_loss = eval_loss / eval_mask_examples
    print("Loss/eval",eval_epoch_loss)
    if config.neural_pretrainer.loss.type == "ctc":
        eval_epoch_PER = eval_errors / eval_phonemes 
        print("PER/eval",eval_epoch_PER)
    elif config.neural_pretrainer.loss.type == "poisson":
        eval_epoch_RMS = (eval_RMS / eval_examples ) ** 0.5
        print("RMS/eval",eval_epoch_RMS)
        eval_epoch_mask_RMS = (eval_mask_RMS / eval_mask_examples) ** 0.5
        print("MaskRMS/eval",eval_epoch_mask_RMS)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for pretrain evaluation", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)