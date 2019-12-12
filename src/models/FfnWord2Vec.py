#  Copyright (c) 2019, Maximilian Pfundstein
#  Please check the attached license file as well as the license notice in the readme.

import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class FfnWord2Vec(nn.Module):
    def __init__(self, word_embeddings, embedding_size=300, padding=200, category_amount=5, dropout=0.25, hidden=256):
        super().__init__()

        # Predefined word embeddings
        self.embedding = nn.Embedding.from_pretrained(word_embeddings)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        self.l1 = nn.Linear(embedding_size * padding, hidden)
        self.l2 = nn.Linear(hidden, category_amount)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.embedding(x)
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.softmax(x)

        return x
