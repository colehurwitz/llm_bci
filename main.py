import argparse
from utils.config_utils import config_from_kwargs, update_config, ParseKwargs

import data_utils.datasets
from data_utils.speechbci_dataset import load_competition_data

from models.trainer import Trainer, default_trainer_config

def main(args):
    config = update_config(default_trainer_config(), args.config_file if args.config_file != "none" else None) 
    config = update_config(config, config_from_kwargs(args.kwargs))

    dataset = load_competition_data(config.data.dataset_dir, **config.data)

    trainer = Trainer(config, dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)