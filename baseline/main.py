import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import argparse
import datetime
import json

from data_loader import DataLoader
from deep.base_model import BaseModel
from deep.text_cnn import TextCNN
from deep.text_rnn import TextRNN
from deep.text_selfattention import Self_Attention
from deep.text_lstm import TextLSTM
from deep.text_transformer import Transformer

# Global variable for storing deep learning models.
dl_model_dict = {
    'mlp': BaseModel,
    'textcnn': TextCNN,
    'textrnn': TextRNN,
    'textselfattention': Self_Attention,
    'textlstm': TextLSTM,
    'texttransformer': Transformer,
}

def main_dl(config):
    data_name = config['data_name']  # aapr
    model_name = config['model_name']  # mlp

    data_loader = DataLoader()

    word_dict_path = "exp/{}/vocab.cover1.min0.json".format(data_name)
    with open(word_dict_path, 'r', encoding='utf-8') as fp:
        word_dict = json.load(fp)
        print("Load word dict from {}.".format(word_dict_path))

    input_path_train = 'datasets/{}/train.input'.format(data_name)
    input_path_val = 'datasets/{}/val.input'.format(data_name)
    input_path_test = 'datasets/{}/test.input'.format(data_name)

    output_path_train = 'datasets/{}/train.output'.format(data_name)
    output_path_val = 'datasets/{}/val.output'.format(data_name)
    output_path_test = 'datasets/{}/test.output'.format(data_name)

    save_folder = 'exp/{}/dl/'.format(data_name)
    # Create a folder if it does not exist.
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    save_model_folder = 'exp/{}/dl/{}/'.format(data_name, model_name)
    # Create a folder if it does not exist.
    if not os.path.exists(save_model_folder):
        os.mkdir(save_model_folder)

    vocab_size = len(word_dict)
    # Instantiate the deep learning model specified by model_name.
    model = dl_model_dict[model_name](vocab_size=vocab_size, **config)

    # Call train_model function to train this model.
    model.train_model(model, data_loader.data_generator, input_path_train, output_path_train, word_dict,
                      input_path_val=input_path_val, output_path_val=output_path_val,
                      input_path_test=input_path_test, output_path_test=output_path_test,
                      save_folder=save_model_folder)

if __name__ == '__main__':
    # Record the start time of the program.
    start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    args = parser.parse_args()

    # Configuration for deep learning model parameters.
    config_path = './config/{}/{}/{}.json'.format(args.phase.strip().split('.')[0],
                                                  args.phase.strip().split('.')[1],
                                                  args.phase.strip())

    if not os.path.exists(config_path):
        raise RuntimeError("There is no {} config.".format(args.phase))

    # Load config parameters using json.
    config = json.load(open(config_path, 'r', encoding='utf-8'))
    print('config: ', config)

    model_name = config['model_name']
    if model_name in dl_model_dict:
        # Run the deep learning main function.
        main_dl(config)
    else:
        raise RuntimeError("There is no model named {}.".format(model_name))

    # Record the end time of the program.
    end_time = datetime.datetime.now()
    # Calculate total run time in seconds.
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))
    print('Done main!')