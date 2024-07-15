import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep.base_model import BaseModel

class TextLSTM(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(TextLSTM, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                       dropout_rate, learning_rate, num_epochs, batch_size,
                                       criterion_name, optimizer_name, gpu, **kwargs)

        # Set the number of LSTM layers, default is 2
        self.num_layers = kwargs.get('num_layers', 2)
        # Directionality of the LSTM, 2 for bidirectional, 1 for unidirectional
        self.num_directions = kwargs.get('num_directions', 2)
        self.bidirection = self.num_directions == 2

        # Activation and parameter initialization
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_dim * 2))

        # LSTM setup
        self.lstm = nn.LSTM(embed_dim, hidden_dim, self.num_layers,
                            batch_first=True, dropout=dropout_rate, bidirectional=self.bidirection)
        # Output layer setup
        self.fc_out = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        # Find sequence length for dynamic batching
        seq_len = torch.argmin(x, dim=1, keepdim=True)
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        # LSTM processing
        hidden, _ = self.lstm(embed)  # [batch_size, seq_len, hidden_dim * num_directions]

        # Pooling over time steps
        hidden = nn.AvgPool2d((hidden.size(1), 1))(hidden)
        hidden = hidden.squeeze(1)

        # Dropout for regularization
        hidden = self.dropout(hidden)
        out = self.fc_out(hidden)  # [batch_size, num_classes]
        return out

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="TextLSTM model training and testing script.")
    parser.add_argument('--phase', default='test', help='Specifies the phase of the model operation.')

    args = parser.parse_args()

    if args.phase == 'test':
        print("This is a test process.")
        # Set up test cases to verify model functionality
        # Initialize model parameters
        vocab_size, embed_dim, hidden_dim, num_classes, \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 100, 64, 64, 2, 0.5, 0.001, 3, 32, 'CrossEntropyLoss', 'Adam', 0
        # Create an instance of the class
        model = TextLSTM(vocab_size, embed_dim, hidden_dim, num_classes,
                         dropout_rate, learning_rate, num_epochs, batch_size,
                         criterion_name, optimizer_name, gpu)
        # Pass simple data to check model output
        input_data = torch.LongTensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 2, 7, 5]])
        output_data = model(input_data)
        print("The output is: {}".format(output_data))

        print("The test process is done.")
    else:
        print("There is no {} function. Please check your command.".format(args.phase))

    end_time = datetime.datetime.now()
    print("{} takes {} seconds.".format(args.phase, (end_time - start_time).seconds))
    print("Done with TextLSTM script.")
