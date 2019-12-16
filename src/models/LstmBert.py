#  Copyright (c) 2019, Maximilian Pfundstein
#  Please check the attached license file as well as the license notice in the readme.

from torch import nn

import logging

logger = logging.getLogger(__name__)


class LstmBert(nn.Module):
    def __init__(self, embedding_size=768, padding=200,
                 category_amount=5, lstm_layers=2, lstm_hidden=128,
                 dropout=0.25, lstm_dropout=0.25):

        super().__init__()

        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.lstm_hidden_states = None

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

        # Decouple from training history
        self.lstm_hidden_states = tuple([each.data for each in self.lstm_hidden_states])

        # noinspection PyTypeChecker
        x, self.lstm_hidden_states = self.lstm(x, self.lstm_hidden_states)
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

        self.lstm_hidden_states = (weight.new(self.lstm_layers, batch_size, self.lstm_hidden).zero_().cuda(),
                                   weight.new(self.lstm_layers, batch_size, self.lstm_hidden).zero_().cuda())
