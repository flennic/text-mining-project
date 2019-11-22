import pandas as pd
import logging
from os import path, remove
import numpy as np
import pickle

logger = logging.getLogger(__name__)


def concatenate_data(settings):

    logger.info("Starting to process original data.")

    if path.exists(settings["orig_output_path"]):
        if settings["use_cache"]:
            logger.info("The original data is already concatenated.")
            return
        else:
            remove(settings["orig_output_path"])

    logger.info("Concatenating data...")

    train = pd.read_csv(settings["orig_train_path"], header=None)
    test = pd.read_csv(settings["orig_test_path"], header=None)

    data = pd.concat([train, test])
    data.columns = ["category", "title", "review"]

    data.to_csv(settings["orig_output_path"], index=False)

    logger.info("Done concatenating data.")


def create_data_sets(settings):

    logger.info("Splitting data...")
    pickle_path = "create_data_sets.pickle"

    if settings["use_cache"] and path.exists(pickle_path) and path.isfile(pickle_path):
        logger.info("Loading data from cache...")
        with open(pickle_path, "rb") as pickled:
            data_sets = pickle.load(pickled)
            logger.info("Done loading data from cache.")
            return zip(*data_sets)

    fractions = np.array([0.4, 0.3, 0.2, 0.1])

    data = pd.read_csv(settings["orig_output_path"])
    data = data.sample(frac=1, random_state=settings["seed"]).reset_index(drop=True)

    train, val, test, stest = np.array_split(data, (fractions[:-1].cumsum() * len(data)).astype(int))

    if settings["use_cache"]:
        logger.info("Creating cache...")
        with open(pickle_path, "wb") as pickled:
            pickle.dump(zip(train, val, test, stest), pickled)

    logger.info("Done splitting data.")

    return train, val, test, stest


