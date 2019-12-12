from torch.utils.data import IterableDataset
import logging

logger = logging.getLogger(__name__)


class AmazonReviewDatasetWord2Vec(IterableDataset):
    def __init__(self, path):

        self._length = 0
        self._path = path

    def __iter__(self):
        return map(self.__process__, open(self._path))

    @property
    def length(self):
        if self._length == 0:
            with open(self._path, 'r') as file:
                for line in file:
                    self._length += 1
        return self._length

    @staticmethod
    def __process__(line):
        return [int(entry) for entry in line.split(",")]
