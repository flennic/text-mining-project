from torch.utils.data import IterableDataset
import logging

logger = logging.getLogger(__name__)


class AmazonReviewDatasetWord2Vec(IterableDataset):
    """Represents the data set 'Amazon Reviews for Sentiment Analysis' found at Kaggle:
    https://www.kaggle.com/bittlingmayer/amazonreviews/kernels
    It's implemented as an iterable data set as the data is quite large and does not necessarily fit to memory."""

    def __init__(self, path):
        """
        Initializes the iterable data set.
        @param path: The path to the file as a string. Absolute or relative.
        """
        self._length = 0
        self._path = path

    def __iter__(self):
        """
        Implement the iterator interface. Returns a new line of the given file, with pre-processing applied as a map.
        @return:
        """
        return map(self.__process__, open(self._path))

    @property
    def length(self):
        """
        Lazy evaluation of the length of the given dataset. Once calculated, the result is stored.
        @return: Returns the length of the dataset as an integer.
        """
        if self._length == 0:
            with open(self._path, 'r') as file:
                for line in file:
                    self._length += 1
        return self._length

    @staticmethod
    def __process__(line):
        """
        Splits a text line by comma and parse all entries to integer. Only for csv files with integers only.
        @param line: A line of text (string).
        @return: Returns a list of all entries parsed to integer.
        """
        return [int(entry) for entry in line.split(",")]
