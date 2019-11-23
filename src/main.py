from json import JSONDecodeError

import utilities
import logging
import json
import numpy as np

# Default settings. Will be overwritten by parameters set in the configuration file.
settings = {
    # Data Processing
    "orig_train_path": u"data/processed/train.csv",
    "orig_test_path": u"data/processed/test.csv",
    "orig_output_path": u"data/processed/data.csv",

    # Models
    "models": {
        "bert": {
            "vocab_size_or_config_json_file": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12
        },
    },

    # Miscellaneous
    "seed": 42,
    "log_level": logging.INFO,
    "use_cache": True
}

# Defining Logging
logging.basicConfig(level=settings["log_level"])
logger = logging.getLogger(__name__)

# Setting a global seed
np.random.seed(settings["seed"])

# Collecting settings
logger.info("Reading settings and updating with default values...")

try:
    with open('settings.json') as settings_file:
        try:
            loaded_settings = json.load(settings_file)
        except JSONDecodeError:
            logger.error("settings.json is an invalid JSON file.")
        settings.update(loaded_settings)
except FileNotFoundError:
    logger.warning("settings.json not found. Proceeding with default values.")
    pass

# Prepare the original data, as it is divided in train and test
utilities.concatenate_data(settings)
train, val, test = utilities.create_data_sets(settings)

# Pre Processing

# Modelling

# Training

# Validating


