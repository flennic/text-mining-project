#  Copyright (c) 2019, Maximilian Pfundstein
#  Please check the attached license file as well as the license notice in the readme.

from json import JSONDecodeError

import torch
import preprocessing
import logging
import json
import numpy as np


# Default settings. Will be overwritten by parameters set in the configuration file.
from interactors.FfnBertModelInteractor import FfnBertModelInteractor
from interactors.FfnWord2VecModelInteractor import FfnWord2VecModelInteractor
from interactors.LstmWord2VecModelInteractor import LstmWord2VecModelInteractor

settings = {
    # Data Processing
    # Paths
    "orig_train_path": "data/original/train.csv",
    "orig_test_path": "data/original/test.csv",
    "processed_data_folder": "data/processed/",
    "cached_model_path": "checkpoints/2019-12-12_22-9_LstmWord2VecModelInteractor.pth",

    # Settings
    "word2vec_path": "data/embeddings/GoogleNews-vectors-negative300.bin",
    "splits": [0.85, 0.1, 0.05],
    "padding": 200,
    "embeddings": 1_000_000,
    "categories": 5,
    "run_model": "lstm_w2v",
    "load_cached_model": False,

    # Models
    "models": {
        "ffn_w2v": {
            "data_loader_workers": 4,
            "batch_size": 8192,
            "learning_rate": 0.05,
            "epochs": 1,
            "embedding_size": 300,  # Fixed for Word2Vec, so do not change
            "dropout": 0.25,
            "hidden": 256
        },
        "lstm_w2v": {
            "data_loader_workers": 2,
            "batch_size": 1024,
            "learning_rate": 0.002,
            "epochs": 2,
            "embedding_size": 300,  # Fixed for Word2Vec, so do not change
            "dropout": 0.25,
            "lstm_layers": 2,
            "lstm_hidden": 128,
            "lstm_dropout": 0.25,
            "gradient_clip": 5
        },
        "ffn_bert": {
            "data_loader_workers": 1,
            "batch_size": 256,
            "learning_rate": 0.5,
            "epochs": 8,
            "embedding_size": 768,  # Fixed for Bert, so do not change
            "dropout": 0.25,
            "hidden": 256,
            "max_batches_per_epoch": 16
        },
        "lstm_bert": {

        }
    },

    # General Parameters
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # Miscellaneous
    "seed": 42,
    "log_level": logging.INFO,
    "cache": True
}

# Defining Logging
logging.basicConfig(level=settings["log_level"])
logger = logging.getLogger(__name__)

# Setting a global seed
np.random.seed(settings["seed"])

# Collecting settings
logger.info("Reading settings and updating with default values.")

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

pre_process_info = preprocessing.preprocess(settings)



# Modelling
if settings["run_model"] == "ffn_w2v":
    if settings["load_cached_model"]:
        model = FfnWord2VecModelInteractor.load(settings["cached_model_path"])
    else:
        model = FfnWord2VecModelInteractor(settings, pre_process_info)
elif settings["run_model"] == "lstm_w2v":
    if settings["load_cached_model"]:
        model = LstmWord2VecModelInteractor.load(settings["cached_model_path"])
    else:
        model = LstmWord2VecModelInteractor(settings, pre_process_info)
elif settings["run_model"] == "ffn_bert":
    if settings["load_cached_model"]:
        model = FfnBertModelInteractor.load(settings["cached_model_path"])
    else:
        model = FfnBertModelInteractor(settings, pre_process_info)
else:
    message = "Model {} is not supported.".format(settings["run_model"])
    logger.critical(message)
    raise ValueError(message)


#model._settings["models"]["lstm_w2v"]["epochs"] = settings["models"]["lstm_w2v"]["epochs"]

# Training
model.train()

# Saving
#model.save()

# Testing


