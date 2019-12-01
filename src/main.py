from json import JSONDecodeError

import torch
import preprocessing
import logging
import json
import numpy as np

from transformers.data.processors.utils import DataProcessor

# Default settings. Will be overwritten by parameters set in the configuration file.
settings = {
    # Data Processing
    #"orig_train_path": u"data/processed/train.csv",
    #"orig_test_path": u"data/processed/test.csv",
    #"orig_output_path": u"data/processed/data.csv",
    "train_path": u"data/processed/train.csv",
    "val_path": u"data/processed/val.csv",
    "test_path": u"data/processed/test.csv",
    "word2vec_path" u"'data/embeddings/GoogleNews-vectors-negative300.bin'"

    # Models
    "models": {
        #"bert": {
        #    "vocab_size_or_config_json_file": 30522,
        #    "hidden_size": 768,
        #    "num_hidden_layers": 12,
        #    "num_attention_heads": 12,
        #    "intermediate_size": 3072,
        #    "hidden_act": "gelu",
        #    "hidden_dropout_prob": 0.1,
        #    "attention_probs_dropout_prob": 0.1,
        #    "max_position_embeddings": 512,
        #    "type_vocab_size": 2,
        #    "initializer_range": 0.02,
        #    "layer_norm_eps": 1e-12
        #},
        "bert": {
            "model": "bert-base-cased",
            "task_name": "task_name",
            "cache_dir": "cache/",
            "sequence_length": 128,
            "train_batch_size": 32,
            "eval_batch_size:": 64,
            "learning_rate": 2e-5,
            "epochs": 2,
        }
    },

    # General Parameters
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

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

# Pre processing
# Prepare the original data, as it is divided in train and test
preprocessing.concatenate_data(settings)
preprocessing.create_data_sets(settings)

# Create embedded data sets
preprocessing.create_word2vec_embeddings(settings)


#processor = DataProcessor()
#train_examples = processor.get_train_examples(DATA_DIR)
#train_examples_len = len(train_examples)

# Modelling

# Training

# Validating


