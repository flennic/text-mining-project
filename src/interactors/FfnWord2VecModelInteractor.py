import os
import torch
import logging
import numpy as np
import preprocessing

from datetime import datetime
from torch.utils.data import DataLoader
from models.FfnWord2Vec import FfnWord2Vec
from datasets.AmazonReviewDatasetWord2Vec import AmazonReviewDatasetWord2Vec

logger = logging.getLogger(__name__)


# noinspection PyTypeChecker,PyProtectedMember,PyArgumentList,DuplicatedCode
class FfnWord2VecModelInteractor:
    """Model interactor for storing and managing the PyTorch model. This one implements a simple feed-forward
    network using word embeddings from Word2Vec for text classification."""

    def __init__(self, settings, info, load_embeddings=True):
        """
        Creates the interactor object and conducts all required steps, so initialization might take a while.
        @param settings: A dictionary that provides all required keys. See example config file for all option.
        @param info: A dictionary that contains the paths to the preprocessed files and optionally the embedded
                        word vectors if already loaded, so that they can be reused.
        @param load_embeddings: If set to false, word vectors are never loaded. If none are available, the embeddings
                        layer will be filled with just zeros. Might come in handy when loading a model where the
                        embeddings are overwritten anyways.
        """

        logger.info("Initializing FnnWord2Vec model interactor.")

        # Saving settings
        self._settings = settings
        self._info = info

        # Training
        self._optimizer = None
        self._criterion = None
        self._no_labels = 5
        self._trained_epochs = 0
        self.train_losses, self.train_accuracies, self.validation_losses, self.validation_accuracies = [], [], [], []

        logger.info("Creating data sets.")

        # Creating data sets
        self._train_data = AmazonReviewDatasetWord2Vec(info["processed_train_file"])
        self._val_data = AmazonReviewDatasetWord2Vec(info["processed_val_file"])
        self._test_data = AmazonReviewDatasetWord2Vec(info["processed_test_file"])

        logger.info("Creating data loaders.")

        # Creating (lazy) data loaders
        self._dataloader_train = DataLoader(self._train_data,
                                            batch_size=settings["models"]["ffn_w2v"]["batch_size"],
                                            num_workers=settings["models"]["ffn_w2v"]["data_loader_workers"],
                                            collate_fn=self.__batch2tensor__)

        self._dataloader_val = DataLoader(self._val_data,
                                          batch_size=settings["models"]["ffn_w2v"]["batch_size"],
                                          num_workers=settings["models"]["ffn_w2v"]["data_loader_workers"],
                                          collate_fn=self.__batch2tensor__)

        self._dataloader_test = DataLoader(self._test_data,
                                           batch_size=settings["models"]["ffn_w2v"]["batch_size"],
                                           num_workers=settings["models"]["ffn_w2v"]["data_loader_workers"],
                                           collate_fn=self.__batch2tensor__)

        logger.info("Creating model.")

        # Creating network
        if info.get("embedded_vectors") is None and load_embeddings:
            logger.info("Embedded vectors not preloaded. Have to load word embeddings for model.")
            tokenizer = preprocessing.get_tokenizer()
            self._info["embedded_vectors"] =\
                preprocessing.get_embedder(settings, tokenizer._unk_token, tokenizer._pad_token).vectors

        if info.get("embedded_vectors") is None and not load_embeddings:
            # noinspection PyUnusedLocal
            tokenizer = None
            self._info["embedded_vectors"] = np.zeros((settings["embeddings"]+2,
                                                       settings["models"]["ffn_w2v"]["embedding_size"]))

        self._model = FfnWord2Vec(
            word_embeddings=torch.FloatTensor(self._info["embedded_vectors"]),
            embedding_size=self._settings["models"]["ffn_w2v"]["embedding_size"],
            padding=self._settings["padding"],
            category_amount=self._settings["categories"],
            dropout=self._settings["models"]["ffn_w2v"]["dropout"])

        # noinspection PyUnresolvedReferences
        self._model = self._model.to(settings["device"])

        # Tokenizer and Embedder not needed any more
        del tokenizer, info["embedded_vectors"]

        logger.info("Model created.")

    # noinspection PyArgumentList
    @staticmethod
    def __batch2tensor__(batch):
        """
        Takes a batch ans transforms it in such a way that it can directly be fed to the network.
        @param batch: List of x and y labels.
        @return: Two tensors, one for x and one for y.
        """
        x, y = [None] * len(batch), [None] * len(batch)
        for i, row in enumerate(batch):
            y[i] = row[0]
            x[i] = row[1:]

        return torch.LongTensor(x), torch.LongTensor(y)

    def train(self):
        """
        Trains the model until the number of epochs in settings for the model is reached. Prints metrics while
        processing. Losses and accuracies are saved within the object.
        """
        logger.info("Beginning training of model (FNN, Word2Vec).")

        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=self._settings["models"]["ffn_w2v"]["learning_rate"])
        self._criterion = torch.nn.NLLLoss()

        while self._trained_epochs < self._settings["models"]["ffn_w2v"]["epochs"]:

            training_loss = 0
            training_accuracy = 0

            for x, y in self._dataloader_train:

                x = x.to(self._settings["device"])
                y = y.to(self._settings["device"])

                # Reset Gradients
                self._optimizer.zero_grad()

                # Forward, Loss, Backwards, Update
                output = self._model(x)
                loss = self._criterion(output, y)
                loss.backward()
                self._optimizer.step()

                # Calculate Metrics
                training_loss += loss.item()
                training_accuracy += torch.sum(torch.exp(output).topk(1)[1].view(-1) == y).item()

            else:

                self._trained_epochs += 1

                validation_loss = 0
                validation_accuracy = 0

                self._model.eval()

                with torch.no_grad():

                    for x, y in self._dataloader_val:

                        x = x.to(self._settings["device"])
                        y = y.to(self._settings["device"])

                        output_validation = self._model(x)
                        loss_val = self._criterion(output_validation, y)
                        validation_loss += loss_val.item()
                        validation_accuracy += torch.sum(
                            torch.exp(output_validation).topk(1, dim=1)[1].view(-1) == y).item()

                training_loss /= (self._train_data.length *
                                  self._settings["models"]["ffn_w2v"]["data_loader_workers"])
                training_accuracy /= (self._train_data.length *
                                      self._settings["models"]["ffn_w2v"]["data_loader_workers"])
                validation_loss /= (self._val_data.length *
                                    self._settings["models"]["ffn_w2v"]["data_loader_workers"])
                validation_accuracy /= (self._val_data.length *
                                        self._settings["models"]["ffn_w2v"]["data_loader_workers"])

                # Saving metrics
                self.train_losses.append(training_loss)
                self.train_accuracies.append(training_accuracy)
                self.validation_losses.append(validation_loss)
                self.validation_accuracies.append(validation_accuracy)

                logger.info("\n\nEpoch: {}/{}\n".format(self._trained_epochs,
                                                        self._settings["models"]["ffn_w2v"]["epochs"]) +
                            "Training Loss: {:.6f}\n".format(training_loss) +
                            "Training Accuracy: {:.3f}\n".format(training_accuracy) +
                            "Validation Loss: {:.6f}\n".format(validation_loss) +
                            "Validation Accuracy: {:.3f}\n".format(validation_accuracy))

                self._model.train()

        logger.info("Training completed.")

    def save(self):
        """
        Saves to model to the path specified in the settings. Subdirectory is always 'checkpoints'.
        @return: None.
        """
        logger.info("Start saving model.")

        checkpoint = {"settings": self._settings,
                      "info": self._info,
                      "trained_epochs": self._trained_epochs,
                      "train_losses": self.train_losses,
                      "train_accuracies": self.train_accuracies,
                      "validation_losses": self.validation_losses,
                      "validation_accuracies": self.validation_accuracies,
                      "state_dict": self._model.state_dict()}

        try:
            os.mkdir("checkpoints")
        except FileExistsError:
            logger.info("Checkpoint folder (checkpoints) already exists.")
        else:
            logger.critical("Checkpoint folder does not exist nor can be created. Abort saving.")
            return

        time = datetime.now()
        model_filename = "checkpoints/{}-{}-{}_{}-{}_{}.pth"\
            .format(time.year, time.month, time.day, time.hour, time.minute, self.__class__.__name__)

        torch.save(checkpoint, model_filename)

        logger.info("Model successfully saved: {}".format(model_filename))

    @staticmethod
    def load(filepath):
        """
        Loads a model. Be aware that the old settings are also reloaded, so you have to manually overwrite settings
        you want to change after reloading.
        @param filepath: Path to the saves checkpoint file.
        @return: Returns a complete model interactor with the model loaded.
        """
        logger.info("Start loading model.")

        checkpoint = torch.load(filepath)

        interactor = FfnWord2VecModelInteractor(checkpoint["settings"], checkpoint["info"], load_embeddings=False)

        interactor._model.load_state_dict(checkpoint['state_dict'])
        interactor._trained_epochs = checkpoint["trained_epochs"]
        interactor.train_losses = checkpoint["train_losses"]
        interactor.train_accuracies = checkpoint["train_accuracies"]
        interactor.validation_losses = checkpoint["validation_losses"]
        interactor.validation_accuracies = checkpoint["validation_accuracies"]

        logger.info("Model successfully loaded from: {}".format(filepath))

        return interactor
