import argparse
import datetime
import torch
torch.cuda.init()
from train import Trainer
from our_model.scibert import SCIBERT
from our_model.onlypro import ONLYPRO
from our_model.onlyref import ONLYREF
from our_model.spsbert import SPSBERT
from our_model.base_model import BaseModel
from our_model.text_cnn import TextCNN
from our_model.text_rnn import TextRNN
from our_model.text_dpcnn import TextDPCNN
from our_model.text_selfattention import Self_Attention
from our_model.text_lstm import TextLSTM
from our_model.text_transformer import Transformer

# Dictionary to map model names to their corresponding class instances with a set dropout rate
dl_model_dict = {
    'scibert': SCIBERT(dropout_rate=0.2),
    'onlyref': ONLYREF(dropout_rate=0.2),
    'onlypro': ONLYPRO(dropout_rate=0.2),
    'spsbert': SPSBERT(dropout_rate=0.2),
    'mlp': BaseModel(dropout_rate=0.2),
    # 'cnn': TextCNN(dropout_rate=0.2),
#    'rnn': TextRNN(dropout_rate=0.2),
#     'dpcnn': TextDPCNN(dropout_rate=0.2),
#     'selfattention': Self_Attention(dropout_rate=0.2),
#     'lstm': TextLSTM(dropout_rate=0.2),
#     'transformer': Transformer(dropout_rate=0.2),
}
print('初始化结束')

def main_dl(args):
    # Extract arguments
    model_name = args.model
    num_epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    data = args.data
    label = args.label

    # Define paths for data (assumes that training, validation, and test data are in one file each)
    data_train_path = f'data_clean/{data}/train_data.json'
    data_val_path = f'data_clean/{data}/val_data.json'
    data_test_path = f'data_clean/{data}/test_data.json'

    save_model_folder = f'model/{data}_{label}_{model_name}'

    # Retrieve model from dictionary
    model = dl_model_dict[model_name]

    # Log the start of training
    print("Begin to train")
    # Initialize trainer with parameters
    trainer = Trainer(
        model=model,
        lr=lr,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs=num_epochs,
        batch_size=batch_size,
        label_selet=label
    )

    # Train model
    trainer.train_model(
        train_data_path=data_train_path,
        val_data_path=data_val_path,
        test_data_path=data_test_path,
        save_folder=save_model_folder
    )

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train a model on scientific texts.')
    parser.add_argument('--model', required=True, help='The name of the model to use.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data', required=True, help='Base name of the data to use.')
    parser.add_argument('--label', required=True, help='Base name of the data to use.')

    args = parser.parse_args()

    # Print arguments for confirmation
    print('Arguments: ', args)

    # Check if the model is in the dictionary and proceed
    if args.model in dl_model_dict:
        main_dl(args)
    else:
        # If model is not found, raise an error
        raise RuntimeError("Model name {} is not recognized.".format(args.model))

    end_time = datetime.datetime.now()
    # Calculate the total time taken
    print('{} takes {} seconds.'.format(args.model, (end_time - start_time).seconds))
    print('Done main!')