import pandas as pd
import logging
import os
import numpy as np
from transformers import BertTokenizer
import multiprocessing as mp
import gensim

logger = logging.getLogger(__name__)

__tokenizer = None
__embedder = None
__pad_token = None
__unk_token = None
__padding = None


# noinspection PyProtectedMember
# noinspection PyTypeChecker,DuplicatedCode,PyUnboundLocalVariable
def preprocess_w2v(settings):
    """
    Pre-processes the original files to indices that can directly be fed to the network. Performs concatenating,
    column-reduction, shuffling, tokenization, padding and vectorization.
    @param settings: A dictionary containing the required keys.
    @return: Returns a dictionary containing the paths of the created files and the loaded word embeddings so that they
            can be used in further processing steps if needed.
    """
    logger.info("Starting to process data for Word2Vec.")

    l_indicator = "w2v"
    l_embedding = settings["embeddings"]
    l_padding = settings["padding"]
    file_suffix = "i-{}_e-{}_p-{}".format(l_indicator, l_embedding, l_padding)

    prep_train_path = "{}train_{}_s{}.csv".format(settings["processed_data_folder"],
                                                  file_suffix, str(settings["splits"][0]))
    prep_val_path = "{}val_{}_s{}.csv".format(settings["processed_data_folder"],
                                              file_suffix, str(settings["splits"][1]))
    prep_test_path = "{}test_{}_s{}.csv".format(settings["processed_data_folder"],
                                                file_suffix, str(settings["splits"][2]))

    if (settings["cache"]
            and os.path.exists(prep_train_path)
            and os.path.exists(prep_val_path)
            and os.path.exists(prep_test_path)):

        logger.info("Word2Vec data already preprocessed. Doing nothing.")

        return {
            "processed_train_file": prep_train_path,
            "processed_val_file": prep_val_path,
            "processed_test_file": prep_test_path,
        }

    logger.info("Reading original data to memory and concatenating them.")

    train = pd.read_csv(settings["orig_train_path"], header=None)
    test = pd.read_csv(settings["orig_test_path"], header=None)
    data = pd.concat([train, test])

    del train, test

    logger.info("Adjusting labels, filtering columns, shuffling and splitting data.")

    data.columns = ["label", "title", "review"]
    data.drop(columns=['title'], inplace=True)
    data["label"] = data["label"] - 1
    data = data.sample(frac=1, random_state=settings["seed"]).reset_index(drop=True)
    train, val, test = np.array_split(data, (np.array(settings["splits"])[:-1].cumsum() * len(data)).astype(int))

    del data

    train = train.values.tolist()
    val = val.values.tolist()
    test = test.values.tolist()

    logger.info("Loading tokenizer and word embeddings.")

    # These must be global, so that they can be pickled for multiprocessing!

    global __padding
    __padding = settings["padding"]

    global __tokenizer
    __tokenizer = get_tokenizer()

    global __embedder
    __embedder = get_embedder(settings, __tokenizer._unk_token, __tokenizer._pad_token)

    global __pad_token, __unk_token
    __pad_token = __tokenizer._pad_token
    __unk_token = __tokenizer._unk_token

    cores = max(1, round(mp.cpu_count() / 2))

    logger.info("Using {} cores for pre-processing.".format(cores))

    # Some repetitive code, but as it breaks easily, this will stay separated. Also the amount of sets will always
    # stay the same

    logger.info("Starting pre-processing for TEST. Might take a while...")

    # Parallelising, will work as long as the processing is not too fast and fills the memory :o
    pool = mp.Pool(cores)
    processed_test = pool.imap(__preprocess, test)

    with open(prep_test_path, "w") as out_file:
        for X, Y in processed_test:
            stringified = [str(entry) for entry in [Y] + X]
            out_file.write(",".join(stringified) + "\n")

    pool.close()
    pool.join()

    del test, processed_test

    logger.info("Starting pre-processing for VAL. Might take a while...")

    # Parallelising, will work as long as the processing is not too fast and fills the memory :o
    pool = mp.Pool(cores)
    processed_val = pool.imap(__preprocess, val)

    with open(prep_val_path, "w") as out_file:
        for X, Y in processed_val:
            stringified = [str(entry) for entry in [Y] + X]
            out_file.write(",".join(stringified) + "\n")

    pool.close()
    pool.join()

    del val, processed_val

    logger.info("Starting pre-processing for TRAIN. Might take a while...")

    # Parallelising, will work as long as the processing is not too fast and fills the memory :o
    pool = mp.Pool(cores)
    processed_train = pool.imap(__preprocess, train)

    with open(prep_train_path, "w") as out_file:
        for X, Y in processed_train:
            stringified = [str(entry) for entry in [Y] + X]
            out_file.write(",".join(stringified) + "\n")

    pool.close()
    pool.join()

    del train, processed_train, pool

    # or return and reuse
    del __embedder, __tokenizer, __pad_token, __unk_token, __padding

    logger.info("Pre-processing completed.")

    return {
        "embedded_vectors": __embedder.vectors,
        "processed_train_file": prep_train_path,
        "processed_val_file": prep_val_path,
        "processed_test_file": prep_test_path,
    }


# noinspection PyTypeChecker
def get_embedder(settings, unk_token, pad_token):
    """
    Creates the word2vec embeddings provided by gensim. Add tokens for padding and unknown. Padding vectors are all 0
    while unknown are the mean of all known vectors.
    @param settings: Dictionary containing the required keys.
    @param unk_token: The token to use for unknown word (string).
    @param pad_token: The token to use for padding (string).
    @return: Returns the embedding model from gensim.
    """
    embedder = gensim.models.KeyedVectors.load_word2vec_format(
        settings["word2vec_path"], limit=settings["embeddings"], binary=True)
    embedder.add(unk_token, np.mean(embedder.vectors, axis=0), replace=False)
    embedder.add(pad_token, np.zeros(300), replace=False)

    return embedder


def get_tokenizer():
    """
    Loads the pre-trained tokenizer 'bert-base-uncased' from the transformers library.
    @return: Returns the tokenizer provided by the transformers library.
    """
    return BertTokenizer.from_pretrained('bert-base-uncased')


def __preprocess(row):
    """
    Pre-processes a row by first tokenizing it, then padding it and finding unknown words. Last indices are obtained.
    @param row: A string of text.
    @return: Returns the indexed sentence and the label.
    """
    # row = [label, review]

    # Tokenize
    sentence = __tokenizer.tokenize(row[1][:__padding])

    # Pad
    # noinspection PyTypeChecker
    sentence = sentence + [__pad_token] * (__padding - len(sentence))

    # Unknown words
    filled_sentence = [word if __embedder.vocab.get(word) is not None else __unk_token for word in sentence]

    # To indices
    sentence_as_int = [__embedder.vocab.get(word).index for word in filled_sentence]

    # X, Y
    return sentence_as_int, row[0]
