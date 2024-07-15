#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data_Processor
======
A class for something.
"""

import os
import sys
import argparse
import datetime

import json
import random
import nltk
import pandas as pd


class DataProcessor(object):
    def __init__(self):
        print('Init...')
        # Path to the dataset.
        self.data_root = './datasets/'
        # Path to the original dataset.
        self.original_root = self.data_root + 'original/'
        self.aapr_root = self.original_root + 'AAPR_Dataset/'

        # Path to the experimental results.
        self.exp_root = './exp/'
        # Path to the logs.
        self.log_root = './logs/'

        # Create directories if they don't exist.
        if not os.path.exists(self.exp_root):
            os.mkdir(self.exp_root)

        if not os.path.exists(self.log_root):
            os.mkdir(self.log_root)

    ###################################
    # Original data processing
    ###################################

    ####################
    # Processing the AAPR dataset
    ####################

    # Show json data
    def show_json_data(self):
        for i in range(4):
            path = self.aapr_root + 'data{}'.format(i+1)
            # Read a json file and get a dictionary variable.
            with open(path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            print(len(data))
            # Iterating through dictionary.
            for paper_id, info in data.items():
                for key, value in info.items():
                    print(key)
                break

    # Extract abstracts (abs) and labels (label) from the AAPR dataset
    def extract_abs_label(self):
        path = self.aapr_root + 'example.csv'
        df = pd.read_csv(path, encoding='utf-8')
        text_list = df['abstract']
        print(len(text_list))
        label_list = df['d']
        print(len(label_list))
        return text_list, label_list

    # Save the extracted abs and label to data.input and data.output.
    def save_abs_label(self):
        save_path = self.data_root + 'aapr/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        abs_list, label_list = self.extract_abs_label()
        input_path = save_path + 'data.input'
        output_path = save_path + 'data.output'
        self.save_pair(data_input=abs_list, data_output=label_list, input_path=input_path, output_path=output_path)

    ###################################
    # General modules
    ###################################

    # Save a list of strings to a file at a given path.
    def save_single(self, data, path):
        count = 0
        list_1 = []
        # When writing data to a file, only str type data can be written.
        with open(path, 'w', encoding='utf-8') as fw:
            for line in data:
                list_1.append(line)
                fw.write(str(line).lower() + '\n')
                count += 1
        print(len(list_1))
        print("Done for saving {} lines to {}.".format(count, path))

    # Save both input and output data simultaneously.
    def save_pair(self, data_input, data_output, input_path, output_path):
        self.save_single(data_input, input_path)
        self.save_single(data_output, output_path)

    # Split data.input/output into three datasets: train.input/output, val.input/output, test.input/output
    def split_data(self, data_name='aapr', split_rate=0.7):
        # Read data from file, resulting in a list.
        with open(self.data_root + '{}/data.input'.format(data_name), 'r', encoding='utf-8') as fp:
            data_input = list(map(lambda x: x.strip(), fp.readlines()))
            print(len(data_input))
            print("Successfully load input data from {}.".format(self.data_root + '{}/data.input'.format(data_name)))

        with open(self.data_root + '{}/data.output'.format(data_name), 'r', encoding='utf-8') as fp:
            data_output = list(map(lambda x: x.strip(), fp.readlines()))
            print(len(data_output))
            print("Successfully load output data from {}.".format(self.data_root + '{}/data.output'.format(data_name)))

        # Shuffle data.
        data = list(zip(data_input, data_output))
        random.shuffle(data)
        data_input, data_output = zip(*data)

        # Determine dataset size.
        data_size = len(data_output)

        # Split original data into training, validation, and testing datasets according to split_rate.
        train_input = data_input[:int(data_size * split_rate)]
        train_output = data_output[:int(data_size * split_rate)]
        val_input = data_input[int(data_size * split_rate): int(data_size * (split_rate + (1 - split_rate) / 2))]
        val_output = data_output[int(data_size * split_rate): int(data_size * (split_rate + (1 - split_rate) / 2))]
        test_input = data_input[int(data_size * (split_rate + (1 - split_rate) / 2)):]
        test_output = data_output[int(data_size * (split_rate + (1 - split_rate) / 2)):]

        # Save split data.
        data_folder = self.data_root + '{}/'.format(data_name)
        self.save_pair(data_input=train_input, data_output=train_output,
                       input_path=data_folder + 'train.input', output_path=data_folder + 'train.output')
        self.save_pair(data_input=val_input, data_output=val_output,
                       input_path=data_folder + 'val.input', output_path=data_folder + 'val.output')
        self.save_pair(data_input=test_input, data_output=test_output,
                       input_path=data_folder + '/test.input', output_path=data_folder + '/test.output')

    # Construct a vocabulary for neural network models
    def get_vocab(self, data_name='aapr', cover_rate=1, mincount=0):
        data_folder = self.data_root + '{}/'.format(data_name)
        # Build the dictionary only from training data.
        train_input_path = data_folder + 'train.input'

        # Count word frequency in the corpus.
        word_count_dict = {}
        total_word_count = 0

        with open(train_input_path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                for word in line.strip().split():
                    total_word_count += 1
                    if word not in word_count_dict:
                        word_count_dict[word] = 1
                    else:
                        word_count_dict[word] += 1

        # Sort words by frequency.
        sorted_word_count_dict = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
        print("There are {} words originally.".format(len(sorted_word_count_dict)))

        # Assign a continuous id to each word, filtering by mincount and cover rate.
        word_dict = {'PAD': 0, 'UNK': 1, 'SOS': 2, 'EOS': 3}
        tmp_word_count = 0
        for word, count in sorted_word_count_dict:
            tmp_word_count += count
            current_rate = tmp_word_count / total_word_count
            if count > mincount and current_rate < cover_rate:
                word_dict[word] = len(word_dict)
        print("There are {} words finally.".format(len(word_dict)))

        # Save the dictionary.
        exp_data_folder = self.exp_root + '{}/'.format(data_name)
        if not os.path.exists(exp_data_folder):
            os.mkdir(exp_data_folder)

        word_dict_path = exp_data_folder + 'vocab.cover{}.min{}.json'.format(cover_rate, mincount)
        with open(word_dict_path, 'w', encoding='utf-8') as fw:
            json.dump(word_dict, fw)
        print("Successfully save word dict to {}.".format(word_dict_path))


# Remember, main is an entry point of a Python script.
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    data_processor = DataProcessor()
    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    elif args.phase == 'show_json_data':
        data_processor.show_json_data()
    elif args.phase == 'extract_abs_label':
        data_processor.extract_abs_label()
    elif args.phase == 'save_abs_label':
        data_processor.save_abs_label()
    elif args.phase == 'split_data':
        data_processor.split_data(data_name='aapr', split_rate=0.7)
    elif args.phase == 'get_vocab':
        data_processor.get_vocab(data_name='aapr', cover_rate=1, mincount=0)
    else:
        print("There is no {} function. Please check your command.".format(args.phase))

    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done data_processor!')
