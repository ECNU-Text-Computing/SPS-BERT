import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.metrics import cal_all
from deep.base_model import BaseModel

class TextCNN(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_filters, filter_sizes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(TextCNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                      dropout_rate, learning_rate, num_epochs, batch_size,
                                      criterion_name, optimizer_name, gpu, **kwargs)

        # Number of convolutional filters.
        self.num_filters = num_filters
        # Sizes of the convolutional filters.
        self.filter_sizes = filter_sizes

        # Additional metrics dimension, can be passed as a keyword argument.
        self.metrics_num = kwargs.get('metrics_num', 4)

        # Define convolution layers.
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, self.embed_dim)) for k in self.filter_sizes])
        # Define dropout layer.
        self.dropout = nn.Dropout(self.dropout_rate)
        # Fully connected layer.
        self.fc1 = nn.Linear(self.num_filters * len(self.filter_sizes), self.hidden_dim)

    def conv_and_pool(self, x, conv):
        """Apply a convolution and a max pooling layer to the input tensor."""
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        """Define the forward pass of the model."""
        embed = self.embedding(x)
        embed = embed.unsqueeze(1)  # Reshape for convolution. [batch_size, 1, seq_len, embedding_dim]
        cnn_out = torch.cat([self.conv_and_pool(embed, conv) for conv in self.convs], 1)
        cnn_out = self.dropout(cnn_out)
        hidden = self.fc1(cnn_out)
        out = self.fc_out(hidden) # [batch_size, num_classes]
        return out

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='TextCNN model training and testing script.')
    parser.add_argument('--phase', default='test', help='Specifies the phase of the model operation.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print("There is no function named '{}'. Please check your command.".format(args.phase))

    end_time = datetime.datetime.now()
    print('The {} phase takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done with TextCNN model script!')
