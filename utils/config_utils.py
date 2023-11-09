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
def update_config(new_config, config, cur, name):

    if isinstance(config, DictConfig):
        for field in config:
            assert field in new_config, f"{cur}{field} is not a field in {name}"
            new_config[field] = update_config(new_config[field], config[field], f"{cur}{field}.", name)
    else:
        new_config = config

    return new_config

def create_and_update_config(default_config_file, config_file = None):

    default_config = DictConfig(yaml.safe_load(open(default_config_file,"r")))
    if config_file is None:
        return default_config

    config = DictConfig(yaml.safe_load(open(config_file,"r")))

    return update_config(default_config, config, "", default_config_file)

