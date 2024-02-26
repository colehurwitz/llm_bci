# llm_bci
LLMs for speech decoding

MODELS:
    -> New model classes can be registered in NAME2MODEL. The model should behave as follows:
        - It is initialised with a config file (in trainer_config.model) + kwargs 
        (in trainer_config.method.model_kwargs). The model config should contain a field "model_class"
        with a registered name in order to instantiate it.
        - Its output should subclass ModelOutput. 
        - It should include a method ``save_checkpoint`` that saves the state dict into a folder, which
        is passed as an argument; and similarly a load checkpoint, that loads the state dict from a folder.

DATASETS:
    -> Raw datasets have different splits. The name of the train and the test split should be specified
    in trainer_config.data.data.train/test_name. Each split is a list of examples, where each example
    is a dict containing the different columns of the dataset. The numeric columns like integers or arrays
    should be in numpy.ndarray format. There can be non-numeric columns like strings or non-ndarrays like
    for metadata
    -> The raw datasets are loaded into a pyTorch Dataset. The class is specified in 
    trainer_config.data.dataset_class. New Datsaets, subclassing SpikingDataset, can be registered in
    NAME2DATASET. Any additional kwargs that are needed for new subclasses of SpikingDatasets will be
    passed from trainer_config.method.dataset_kwargs.
    -> The Trainer can be passed a raw dataset, but also the name of a hf dataset or the path to a json
    file containing the dataset. These two ways of loading can also be accessed from 
    trainer_config.data.hf/json_dataset_name.

DATALOADER:
    -> A pyTorch DataLoader with custom bacthing (collate) function is used. The batching function converts numpy
    arrays into torch tensors and adds padding following the config for each column in 
    trainer_config.method.dataloader_kwargs.pad_dict.
    -> TODO: Add custom batching functions, or add functionality to the current one according to new 
    needs.

