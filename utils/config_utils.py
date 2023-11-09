import yaml

class DictConfig(dict):

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictConfig(value)
        return value

    def get_dict(self):
        return super()


# Recursively update the entries of the config dict
def update_config_rec(new_config, config, cur, name):

    if isinstance(config, DictConfig):
        for field in config:
            if not field in new_config:
                print(f"Added {cur}{field} to {name}")
                new_config[field] = DictConfig({})
            new_config[field] = update_config_rec(new_config[field], config[field], f"{cur}{field}.", name)
    else:
        new_config = config

    return new_config


# Update values in default_config from config, adding the missing keys if needed. 
# If config is None, the default config is returned.
# Configs can also be a path to the config file.
def update_config(default_config, config = None, name="config"):

    if isinstance(default_config, str):
        default_config = DictConfig(yaml.safe_load(open(default_config_file,"r")))

    if config_file is None:
        return default_config

    if isinstance(config, str):
        config = DictConfig(yaml.safe_load(open(config_file,"r")))

    return update_config_rec(default_config, config, "", name)

