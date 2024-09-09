import sys
import argparse
import datetime
import torch
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from deep.base_model import BaseModel
from utils.metrics import cal_all


class Self_Attention(BaseModel):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,dim_k, dim_v,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Self_Attention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.q = nn.Linear(embed_dim, dim_k)
        self.k = nn.Linear(embed_dim, dim_k)
        self.v = nn.Linear(embed_dim, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
        self.fc = nn.Linear(dim_v, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        atten = nn.Softmax(dim=-1)(
            torch.bmm(Q, K.permute(0, 2, 1
                                   ))) * self._norm_fact  # Q * K.T()
        output = torch.bmm(atten, V)
        output = torch.sum(output, 1)
        output = F.relu(output)
        output = self.fc(output)
        return output


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    # python3 text_selfattention.py --phase test
    if args.phase == 'test':
        # For testing our model, we can set the hyper-parameters as follows.
        vocab_size, embed_dim, hidden_dim, num_classes, dim_k, dim_v,\
        dropout_rate, learning_rate, num_epochs, batch_size,\
        criterion_name, optimizer_name, gpu = 200, 128, 64, 2, 12,8,0.5, 0.0001, 3, 128, 'CrossEntropyLoss', 'Adam', 0

        # new an objective.
        model = Self_Attention(vocab_size, embed_dim, hidden_dim, num_classes,dim_k, dim_v,
                            dropout_rate, learning_rate, num_epochs, batch_size,
                         criterion_name, optimizer_name, gpu)
        # a simple example of the input.
        input = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
        input = torch.LongTensor(input)  # input: [batch_size, seq_len] = [3, 5]

        # the designed model can produce an output.
        output= model(input)
        print(output)
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')



