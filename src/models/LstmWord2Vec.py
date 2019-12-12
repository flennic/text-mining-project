#  Copyright (c) 2019, Maximilian Pfundstein
#  Please check the attached license file as well as the license notice in the readme.

from torch import nn

import logging

logger = logging.getLogger(__name__)


class LstmWord2Vec(nn.Module):
    def __init__(self, word_embeddings, embedding_size=300, padding=200,
                 category_amount=5, lstm_layers=2, lstm_hidden=128,
                 dropout=0.25, lstm_dropout=0.25):

        super().__init__()

        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden

        # Predefined word embeddings
        self.embedding = nn.Embedding.from_pretrained(word_embeddings)

        # LSTM
        self.lstm = nn.LSTM(embedding_size, lstm_hidden, lstm_layers,
                            batch_first=True, dropout=lstm_dropout,
                            bidirectional=False)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # FFN
        self.l1 = nn.Linear(lstm_hidden * padding, category_amount)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        hidden = self.init_hidden(x.shape[0])

        # Pass the input tensor through each of our operations
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.l1(x)
        x = self.sigmoid(x)
        x = self.softmax(x)

        return x

    # noinspection PyUnresolvedReferences
    def init_hidden(self, batch_size):
        # Create two new tensors with sizes lstm_layers x batch_size x lstm_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_hidden).zero_().cuda(),
                  weight.new(self.lstm_layers, batch_size, self.lstm_hidden).zero_().cuda())

        return hidden
