import pandas as pd
import logging
from os import path, remove
import numpy as np
import spacy
import pickle
import gc
from gensim.models import KeyedVectors

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
    logger.info("Splitting data and creating creating data sets...")
    # pickle_path = "create_data_sets.pickle"

    # if settings["use_cache"] and path.exists(pickle_path) and path.isfile(pickle_path):
    #    logger.info("Loading data from cache...")
    #    with open(pickle_path, "rb") as pickled:
    #        data_sets = pickle.load(pickled)
    #        logger.info("Done loading data from cache.")
    #        return zip(*data_sets)

    if (settings["use_cache"]
            and path.exists(settings["train_path"])
            and path.exists(settings["val_path"])
            and path.exists(settings["test_path"])):
        logger.info("Data sets already created.")
        return

    fractions = np.array([0.5, 0.3, 0.2])

    data = pd.read_csv(settings["orig_output_path"])
    data.rename(columns={"category": "label"}, inplace=True)
    data["label"] = data["label"] - 1
    data.drop("title", axis=1, inplace=True)
    # BERT needs this. Be nice to BERT!
    data.insert(loc=1, column='alpha', value="a")

    data = data.sample(frac=1, random_state=settings["seed"]).reset_index(drop=True)

    train, val, test = np.array_split(data, (fractions[:-1].cumsum() * len(data)).astype(int))

    train.to_csv(settings["train_path"], index=True, index_label="id", header=True)
    val.to_csv(settings["val_path"], index=True, index_label="id", header=True)
    test.to_csv(settings["test_path"], index=True, index_label="id", header=True)

    # if settings["use_cache"]:
    #    logger.info("Creating cache...")
    #    with open(pickle_path, "wb") as pickled:
    #        pickle.dump(zip(train, val, test), pickled)

    logger.info("Done splitting data.")

    return


def create_word2vec_embeddings(settings):
    # Tokenizer
    nlp = spacy.load("en_core_web_sm")
    # Vectorizer
    model = KeyedVectors.load_word2vec_format('data/embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    vector_size = model['hello'].shape[0]

    data_set_paths = (settings["train_path"], settings["val_path"], settings["test_path"])

    for data_set_path in data_set_paths:
        data = pd.read_csv(data_set_path)

        vectorized_data = []

        # id, label, alpha, review
        for index, row in data.iterrows():
            # lemmatized_review = [word.lemma_ for word in nlp(row["review"])]
            # lemmatized_review = [token.lemma_ for token in nlp(row["review"])
            #                     if not token.is_stop and str.isalpha(token.lemma_) and len(token.lemma_) > 2]
            vectorized_review = tuple(model[token.lemma_] if token.lemma_ != "-PRON-" else token.text.lower()
                                      for token in nlp(row["review"])
                                      if token.lemma_ in model.vocab)
            vectorized_data.append((row["id"], row["label"], row["alpha"]) + vectorized_review)

        column_names = ['id', 'label', 'alpha'] + [str(x) for x in range(vector_size + 1)]
        df = pd.DataFrame(vectorized_data, columns=column_names)
        pickle.dump(df, data_set_path + ".vectorized..pickle")
        del df
        gc.collect()
