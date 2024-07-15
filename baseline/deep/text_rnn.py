import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
from deep.base_model import BaseModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

class TextRNN(nn.Module):
    def __init__(self, dropout_rate, **kwargs):
        super(TextRNN, self).__init__(**kwargs)
        # Initialize the RNN model's parameters.
        # Number of RNN layers, default is 2
        self.num_layers = 2
        # Determine if the RNN is bidirectional, default is bidirectional
        self.num_directions = 2
        self.bidirection = self.num_directions == 2

        # Setup the RNN layer
        self.rnn = nn.RNN(embed_dim, hidden_dim, self.num_layers, batch_first=True,
                          dropout=dropout_rate, bidirectional=self.bidirection)
        # Setup the output layer
        self.fc_out = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        # Forward pass of the model.
        # Identify sequence length for dynamic batching.
        seq_len = torch.argmin(x, dim=1, keepdim=True)
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        hidden, _ = self.rnn(embed)  # [batch_size, seq_len, hidden_dim * num_directions]
        # Applying dropout to the last layer of the RNN output before the fully connected layer.
        hidden = self.dropout(hidden[:, seq_len[0][0]-1, :])
        out = self.fc_out(hidden)  # [batch_size, num_classes]
        return out

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="Script for training and testing the TextRNN model.")
    parser.add_argument('--phase', default='test', help='Specifies the phase of the model operation.')

    args = parser.parse_args()

    if args.phase == 'test':
        print("This is a test process.")
        # Setup model parameters for testing.
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 100, 64, 32, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0
        # Initialize the model
        model = TextRNN(dropout_rate, vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                        num_classes=num_classes, learning_rate=learning_rate, num_epochs=num_epochs,
                        batch_size=batch_size, criterion_name=criterion_name, optimizer_name=optimizer_name, gpu=gpu)
        # Test model with simple data input
        input_data = torch.LongTensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
        output_data = model(input_data)
        print("The output is: {}".format(output_data))
        print("The test process is done.")
    else:
        print("There is no {} function. Please check your command.".format(args.phase))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))
    print('Done with TextRNN.')
