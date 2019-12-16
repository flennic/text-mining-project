#  Copyright (c) 2019, Maximilian Pfundstein
#  Please check the attached license file as well as the license notice in the readme.

import os
import torch
import logging

from datetime import datetime
from transformers import DistilBertModel
from models.FfnBert import FfnBert
from torch.utils.data import DataLoader
from datasets.AmazonReviewDatasetWord2Vec import AmazonReviewDatasetWord2Vec

logger = logging.getLogger(__name__)


# noinspection PyTypeChecker,PyProtectedMember,PyArgumentList,DuplicatedCode
class FfnBertModelInteractor:
    """Model interactor for storing and managing the PyTorch model. This one implements a simple feed-forward
    network and expects bert embeddings for text classification."""

    def __init__(self, settings, info):
        """
        Creates the interactor object and conducts all required steps, so initialization might take a while.
        @param settings: A dictionary that provides all required keys. See example config file for all option.
        @param info: A dictionary that contains the paths to the preprocessed files and optionally the embedded
                        word vectors if already loaded, so that they can be reused.
        """

        logger.info("Initializing FnnBert model interactor.")

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
                                            batch_size=settings["models"]["ffn_bert"]["batch_size"],
                                            num_workers=settings["models"]["ffn_bert"]["data_loader_workers"],
                                            collate_fn=self.__batch2tensor__)

        self._dataloader_val = DataLoader(self._val_data,
                                          batch_size=settings["models"]["ffn_bert"]["batch_size"],
                                          num_workers=settings["models"]["ffn_bert"]["data_loader_workers"],
                                          collate_fn=self.__batch2tensor__)

        self._dataloader_test = DataLoader(self._test_data,
                                           batch_size=settings["models"]["ffn_bert"]["batch_size"],
                                           num_workers=settings["models"]["ffn_bert"]["data_loader_workers"],
                                           collate_fn=self.__batch2tensor__)

        logger.info("Creating model.")

        # Loading Bert
        self._bert_model = DistilBertModel.from_pretrained('bert-base-uncased').cuda()
        self._bert_model = self._bert_model.to(settings["device"])

        # Creating network
        self._model = FfnBert(
            embedding_size=self._settings["models"]["ffn_bert"]["embedding_size"],
            padding=self._settings["padding"],
            category_amount=self._settings["categories"],
            dropout=self._settings["models"]["ffn_bert"]["dropout"],
            hidden=self._settings["models"]["ffn_bert"]["hidden"]
        )

        # noinspection PyUnresolvedReferences
        self._model = self._model.to(settings["device"])

        # Tokenizer and Embedder not needed any more
        #del tokenizer, info["embedded_vectors"]

        logger.info("Model created.")

    # noinspection PyArgumentList
    @staticmethod
    def __batch2tensor__(batch):
        """
        Takes a batch and transforms it in such a way that it can directly be fed to the network.
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

        logger.info("Beginning training of model (FNN, Bert).")

        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=self._settings["models"]["ffn_bert"]["learning_rate"])
        self._criterion = torch.nn.NLLLoss()

        while self._trained_epochs < self._settings["models"]["ffn_bert"]["epochs"]:

            training_loss = 0
            training_accuracy = 0

            no_processed_train = 0

            for x, y in self._dataloader_train:

                no_processed_train += 1

                x = x.to(self._settings["device"])
                y = y.to(self._settings["device"])

                with torch.no_grad():
                    x = self._bert_model(x)[0]

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

                #no_processed_train = no_processed_train + self._settings["models"]["ffn_bert"]["batch_size"]
                #print(no_processed / self._train_data.length)
                print(no_processed_train)
                if no_processed_train >= self._settings["models"]["ffn_bert"]["max_batches_per_epoch"]:
                    break

            print(training_accuracy / (self._settings["models"]["ffn_bert"]["batch_size"] * no_processed_train))

            #else:

            self._trained_epochs += 1

            validation_loss = 0
            validation_accuracy = 0

            self._model.eval()

            with torch.no_grad():

                no_processed_val = 0

                for x, y in self._dataloader_val:

                    no_processed_val += 1

                    x = x.to(self._settings["device"])
                    y = y.to(self._settings["device"])

                    x = self._bert_model(x)[0]

                    output_validation = self._model(x)
                    loss_val = self._criterion(output_validation, y)
                    validation_loss += loss_val.item()
                    validation_accuracy += torch.sum(
                        torch.exp(output_validation).topk(1, dim=1)[1].view(-1) == y).item()

                    #no_processed_val = no_processed_val + self._settings["models"]["ffn_bert"]["batch_size"]

                    print(no_processed_val)

                    if no_processed_val >= self._settings["models"]["ffn_bert"]["max_batches_per_epoch"]:
                        break

                print(validation_accuracy / (self._settings["models"]["ffn_bert"]["batch_size"] * no_processed_val))

                #training_loss /= (self._train_data.length *
                #                  self._settings["models"]["ffn_bert"]["data_loader_workers"])
                #training_accuracy /= (self._train_data.length *
                #                      self._settings["models"]["ffn_bert"]["data_loader_workers"])
                #validation_loss /= (self._val_data.length *
                #                    self._settings["models"]["ffn_bert"]["data_loader_workers"])
                #validation_accuracy /= (self._val_data.length *
                #                        self._settings["models"]["ffn_bert"]["data_loader_workers"])

                training_loss /= (no_processed_train * self._settings["models"]["ffn_bert"]["batch_size"])
                training_accuracy /= (no_processed_train * self._settings["models"]["ffn_bert"]["batch_size"])
                validation_loss /= (no_processed_val * self._settings["models"]["ffn_bert"]["batch_size"])
                validation_accuracy /= (no_processed_val * self._settings["models"]["ffn_bert"]["batch_size"])

                # Saving metrics
                self.train_losses.append(training_loss)
                self.train_accuracies.append(training_accuracy)
                self.validation_losses.append(validation_loss)
                self.validation_accuracies.append(validation_accuracy)

                logger.info("\n\nEpoch: {}/{}\n".format(self._trained_epochs,
                                                        self._settings["models"]["ffn_bert"]["epochs"]) +
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
