import random
import sys
import argparse
import datetime

import torch

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from data_processor import DataProcessor
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class DataLoader(DataProcessor):
    def __init__(self):
        super(DataLoader, self).__init__()

    def data_generator(self, input_path, output_path,
                       word_dict=None, batch_size=64, shuffle=True):

        with open(input_path, 'r',encoding='utf-8') as fp:
            input_data = fp.readlines()

        with open(output_path, 'r',encoding='utf-8') as fp:
            output_data = fp.readlines()

        if shuffle:
            data = list(zip(input_data, output_data))
            random.shuffle(data)
            input_data, output_data = zip(*data)

        for i in range(0, len(output_data), batch_size):
            batch_input = []
            for line in input_data[i: i+batch_size]:
                new_line = []
                for word in line.strip().split():
                    new_line.append(int(word_dict[word]) if word in word_dict else int(word_dict['UNK']))
                batch_input.append(torch.Tensor(new_line))
            batch_x = pad_sequence(batch_input, batch_first=True).detach().numpy()
            batch_output = [float(label.strip()) for label in output_data[i: i+batch_size]]
            batch_y = np.array(batch_output)

            yield batch_x, batch_y
