import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from deep.base_model import BaseModel
from torch.autograd import Variable
import copy

class Transformer(BaseModel):
    """
    Transformer model that processes input through embedding and position encoding,
    followed by a sequence of encoder layers, and then outputs through a linear layer
    after aggregation and normalization.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_head, num_encoder,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, **kwargs):
        super(Transformer, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                          dropout_rate, learning_rate, num_epochs, batch_size,
                                          criterion_name, optimizer_name, gpu, **kwargs)

        self.num_encoder = num_encoder
        self.num_head = num_head
        self.position_embedding = Positional_Encoding(vocab_size, embed_dim, hidden_dim, num_classes,
                                                      dropout_rate, learning_rate, num_epochs, batch_size,
                                                      criterion_name, optimizer_name, gpu)
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_classes, num_head,
                               dropout_rate, learning_rate, num_epochs, batch_size,
                               criterion_name, optimizer_name, gpu)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(num_encoder)])
        self.fc1 = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = self.position_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = torch.mean(out, 1)
        out = self.fc1(out)
        return out

class Encoder(BaseModel):
    """ Encoder layer that includes a multi-head attention mechanism and a position-wise feed-forward network. """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_head,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, **kwargs):
        super(Encoder, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                      dropout_rate, learning_rate, num_epochs, batch_size,
                                      criterion_name, optimizer_name, gpu, **kwargs)
        self.num_head = num_head
        self.attention = Multi_Head_Attention(vocab_size, embed_dim, hidden_dim, num_classes, num_head,
                                              dropout_rate, learning_rate, num_epochs, batch_size,
                                              criterion_name, optimizer_name, gpu)
        self.feed_forward = Position_wise_Feed_Forward(vocab_size, embed_dim, hidden_dim, num_classes,
                                                      dropout_rate, learning_rate, num_epochs, batch_size,
                                                      criterion_name, optimizer_name, gpu)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class Positional_Encoding(BaseModel):
    """ Positional Encoding layer using sine and cosine functions of different frequencies. """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, max_len=500, **kwargs):
        super(Positional_Encoding, self).__init__(vocab_size, embed_dim, num_classes, hidden_dim,
                                                  dropout_rate, learning_rate, num_epochs, batch_size,
                                                  criterion_name, optimizer_name, gpu, **kwargs)
        self.max_len = max_len
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class Scaled_Dot_Product_Attention(BaseModel):
    """ Scaled Dot-Product Attention mechanism. """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, **kwargs):
        super(Scaled_Dot_Product_Attention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                                           dropout_rate, learning_rate, num_epochs, batch_size,
                                                           criterion_name, optimizer_name, gpu, **kwargs)

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(BaseModel):
    """ Multi-Head Attention mechanism to facilitate the model attending to information at different positions. """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_head,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, **kwargs):
        super(Multi_Head_Attention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                                   dropout_rate, learning_rate, num_epochs, batch_size,
                                                   criterion_name, optimizer_name, gpu, **kwargs)
        self.num_head = num_head
        assert embed_dim % num_head == 0
        self.dim_head = embed_dim // self.num_head
        self.fc_Q = nn.Linear(embed_dim, num_head * self.dim_head)
        self.fc_K = nn.Linear(embed_dim, num_head * self.dim_head)
        self.fc_V = nn.Linear(embed_dim, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention(vocab_size, embed_dim, hidden_dim, num_classes,
                                                      dropout_rate, learning_rate, num_epochs, batch_size,
                                                      criterion_name, optimizer_name, gpu)
        self.fc = nn.Linear(num_head * self.dim_head, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Position_wise_Feed_Forward(BaseModel):
    """ Position-wise Feed-Forward Network that applies two linear transformations with a ReLU activation in between. """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, **kwargs):
        super(Position_wise_Feed_Forward, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                                         dropout_rate, learning_rate, num_epochs, batch_size,
                                                         criterion_name, optimizer_name, gpu, **kwargs)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Run Transformer model training or testing.')
    parser.add_argument('--phase', default='test', help='Specifies the phase of the model operation.')

    args = parser.parse_args()

    if args.phase == 'test':
        print("This is a test process.")
        vocab_size, embed_dim, hidden_dim, num_classes, num_head, num_encoder,\
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 200, 128, 64, 2, 16, 6, 0.5, 0.0001, 3, 128, 'CrossEntropyLoss', 'Adam', 0

        model = Transformer(vocab_size, embed_dim, hidden_dim, num_classes, num_head, num_encoder,
                            dropout_rate, learning_rate, num_epochs, batch_size,
                            criterion_name, optimizer_name, gpu)
        input_data = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
        output = model(input_data)
        print(output)
    else:
        print("There is no function named '{}'. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))
    print('Done Base_Model!')
