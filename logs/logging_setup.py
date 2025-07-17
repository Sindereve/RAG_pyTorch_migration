import logging.config, yaml

def configure_logging(cfg_path: str = "logs/logging.yaml"):
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
