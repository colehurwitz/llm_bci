## YAML CONFIG FILES ## 
import yaml

""" Wrapper for config dict to access it using dot notation
"""
class DictConfig(dict):

    def __getattr__(self, name):
        # To avoid an exception in case the field is not initialised but it's called
        if name not in self.keys():
            return None

        value = self[name]
        if isinstance(value, dict):
            value = DictConfig(value)
        return value

    def get_dict(self):
        return super()



"""" Recursively update the entries of the config dict, unpacking the include files
"""

def update_config_rec(new_config, config):
    
    # Unpack includes
    if isinstance(config, str) and config.split(":")[0] == "include":
        config = yaml.safe_load(open(config.split(":")[1],"r"))
    if isinstance(new_config, str) and new_config.split(":")[0] == "include":
        new_config = yaml.safe_load(open(new_config.split(":")[1],"r"))

    
    if isinstance(config, dict):
        # Deal with configs that should be a dict but are null
        if new_config is None:
            new_config = {}
        for field in config:
            if not field in new_config:
                new_config[field] = {}
            new_config[field] = update_config_rec(new_config[field], config[field])
    else:
        # Assign leaf
        new_config = config

    return new_config


""" Update values in default_config from config, adding the missing keys if needed. 
    If config is None, the default config is returned (with all the includes unpacked).
    Configs can also be a path to the config file.
"""
def update_config(default_config, config = None):

    if isinstance(default_config, str):
        default_config = yaml.safe_load(open(default_config,"r"))

    # If no config is provided, we iterate using the same config to make sure that the includes
    # are unpacked
    config = default_config if config is None else config

    if isinstance(config, str):
        config = yaml.safe_load(open(config,"r"))

    return DictConfig(update_config_rec(default_config, config))



## COMMAND LINE KWARGS ##
import argparse

""" Parse command line kwargs to dict
"""
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


""" Parse flat kwargs dict with dot notation keys to nested dict
"""
def config_from_kwargs(kwargs):
    
    config = {}
    
    if kwargs is not None:
        for key, value in kwargs.items():
            # Go iteratively to the leaf
            cur = config
            for sub_key in key.split(".")[:-1]:
                if not sub_key in cur:
                    cur[sub_key] = {}
                cur = cur[sub_key]
            cur[key.split(".")[-1]] = value

    return DictConfig(config)

