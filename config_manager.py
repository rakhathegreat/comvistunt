import yaml
import os

CONFIG_PATH = "config.yaml"

def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}

def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def get_config(key: str, default=None):
    config = load_config()
    return config.get(key, default)

def set_config(key: str, value):
    config = load_config()
    config[key] = value
    save_config(config)
