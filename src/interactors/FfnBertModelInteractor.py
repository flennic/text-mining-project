#  Copyright (c) 2019, Maximilian Pfundstein
#  Please check the attached license file as well as the license notice in the readme.

import os
import json
import torch
import logging

from datetime import datetime
from models.FfnBert import FfnBert
from torch.utils.data import DataLoader
from transformers import DistilBertModel
from datasets.AmazonReviewDataset import AmazonReviewDataset

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

        # Metrics
        self._optimizer = None
        self._criterion = None
        self._no_labels = 5
        self._trained_epochs = 0
        self.train_losses, self.train_accuracies, self.validation_losses, self.validation_accuracies = [], [], [], []
        self.test_loss, self.test_accuracy = None, None
        self.trailing_training_accuracy = {}

        logger.info("Creating data sets.")

        # Creating data sets
        self._train_data = AmazonReviewDataset(info["processed_train_file"])
        self._val_data = AmazonReviewDataset(info["processed_val_file"])
        self._test_data = AmazonReviewDataset(info["processed_test_file"])

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
        self._bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
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

        logger.info("Model created.")

    def __repr__(self):
        return self.__combined_representation__()

    def __str__(self):
        return self.__combined_representation__()

    def __combined_representation__(self):

        model_type = self.__class__.__name__
        epochs = self._trained_epochs
        validation_accuracies = self.train_accuracies
        test_accuracy = self.test_accuracy
        model_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        settings = self._settings
        settings["device"] = None
        settings = json.dumps(settings)
        trailing_training_accuracies = json.dumps(self.trailing_training_accuracy)

        return f"\n###################\n" +\
               f"{model_type}\n" +\
               f"Epochs: {epochs}\n" +\
               f"Validation Accuracies: {validation_accuracies}\n" + \
               f"Test Accuracy: {test_accuracy}\n" +\
               f"Model Parameters: {model_parameters}\n" +\
               f"###################\n" +\
               f"Settings: {settings}\n" +\
               f"Trailing Training Accuracies: {trailing_training_accuracies}\n" +\
               f"###################\n"

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
            processed_batches = 0

            for x, y in self._dataloader_train:

                processed_batches += 1

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

                # Print metrics at each 1/info_density step
                info_density = 100
                batch_size = self._settings["models"]["ffn_bert"]["batch_size"]
                epochs = self._settings["models"]["ffn_bert"]["epochs"]
                data_loaders = self._settings["models"]["ffn_bert"]["data_loader_workers"]
                percentage = round(100 * processed_batches * batch_size / self._train_data.length / data_loaders, 1)

                if processed_batches % round(self._train_data.length / info_density / batch_size * data_loaders) == 0:
                    time = datetime.now()
                    logger.info("\n\nEpoch: {}/{} - {}%\n".format(self._trained_epochs, epochs, percentage) +
                                "Training Loss: {:.6f}\n".format(training_loss / (processed_batches * batch_size)) +
                                "Training Accuracy: {:.3f}\n".format(
                                    training_accuracy / (processed_batches * batch_size)) +
                                "Time: {:04d}-{:02d}-{:02d} {:02d}:{:02d}".format(time.year, time.month, time.day,
                                                                                  time.hour, time.minute))

                    # Save trailing accuracy
                    if (self._trained_epochs + 1) not in self.trailing_training_accuracy:
                        self.trailing_training_accuracy[(self._trained_epochs + 1)] = {}

                    self.trailing_training_accuracy[(self._trained_epochs + 1)][percentage] = training_accuracy / (
                            processed_batches * batch_size)

            else:

                logger.info("Evaluating on validation set.")

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

                    training_loss /= (self._train_data.length *
                                      self._settings["models"]["ffn_bert"]["data_loader_workers"])
                    training_accuracy /= (self._train_data.length *
                                          self._settings["models"]["ffn_bert"]["data_loader_workers"])
                    validation_loss /= (self._val_data.length *
                                        self._settings["models"]["ffn_bert"]["data_loader_workers"])
                    validation_accuracy /= (self._val_data.length *
                                            self._settings["models"]["ffn_bert"]["data_loader_workers"])

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

                    self.save()

        logger.info("Training completed.")

    def evaluate(self):
        """
        Evaluates on the test data set and calculates loss and accuracy.
        """

        logger.info("Evaluating on test set.")

        test_loss = 0
        test_accuracy = 0

        self._model.eval()

        with torch.no_grad():
            for x, y in self._dataloader_test:
                x = x.to(self._settings["device"])
                y = y.to(self._settings["device"])

                x = self._bert_model(x)[0]

                output_validation = self._model(x)
                loss_val = self._criterion(output_validation, y)
                test_loss += loss_val.item()
                test_accuracy += torch.sum(torch.exp(output_validation).topk(1, dim=1)[1].view(-1) == y).item()

        test_loss /= (self._test_data.length * self._settings["models"]["ffn_bert"]["data_loader_workers"])
        test_accuracy /= (self._test_data.length * self._settings["models"]["ffn_bert"]["data_loader_workers"])

        # Saving metrics
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy

        # Printing some information after each epoch
        time = datetime.now()
        logger.info("\n\nEpoch: {}/{}\n".format(self._trained_epochs,
                                                self._settings["models"]["ffn_bert"]["epochs"]) +
                    "Test Loss: {:.6f}\n".format(test_loss) +
                    "Test Accuracy: {:.3f}\n".format(test_accuracy) +
                    "Time: {:04d}-{:02d}-{:02d} {:02d}:{:02d}".format(time.year, time.month, time.day, time.hour, time.minute))

        self._model.train()

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
                      "test_loss": self.test_loss,
                      "test_accuracy": self.test_accuracy,
                      "trailing_training_accuracies": self.trailing_training_accuracy,
                      "state_dict": self._model.state_dict()}

        try:
            os.mkdir("checkpoints")
        except FileExistsError:
            logger.info("Checkpoint folder (checkpoints) already exists.")

        time = datetime.now()
        model_filename = "checkpoints/{}-{}-{}_{}-{}_{}.pth" \
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

        interactor = FfnBertModelInteractor(checkpoint["settings"], checkpoint["info"])

        interactor._model.load_state_dict(checkpoint['state_dict'])
        interactor._trained_epochs = checkpoint["trained_epochs"]
        interactor.train_losses = checkpoint["train_losses"]
        interactor.train_accuracies = checkpoint["train_accuracies"]
        interactor.validation_losses = checkpoint["validation_losses"]
        interactor.validation_accuracies = checkpoint["validation_accuracies"]
        interactor.test_loss = checkpoint["test_loss"]
        interactor.test_accuracy = checkpoint["test_accuracy"]
        interactor.trailing_training_accuracy = checkpoint["trailing_training_accuracies"]

        logger.info("Model successfully loaded from: {}".format(filepath))

        return interactor
