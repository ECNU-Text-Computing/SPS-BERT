import sys
import time
import argparse
import datetime
import torch
import torch.nn as nn
import numpy as np
from utils.metrics import cal_all

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(BaseModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion_name = criterion_name
        self.optimizer_name = optimizer_name

        self.embedding = nn.Embedding(self.vocab_size, embed_dim, _weight=None)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1) # for regression, only one output value is needed
        self.dropout = nn.Dropout(dropout_rate)

        self.criterion_dict = {
            'MSELoss': torch.nn.MSELoss,  # Mean Squared Error Loss for regression
            'L1Loss': torch.nn.L1Loss  # Mean Absolute Error Loss for regression
        }
        self.optimizer_dict = {
            'Adam': torch.optim.Adam
        }

        if criterion_name not in self.criterion_dict:
            raise ValueError("There is no criterion_name: {}.".format(criterion_name))
        self.criterion = self.criterion_dict[criterion_name]()

        if optimizer_name not in self.optimizer_dict:
            raise ValueError("There is no optimizer_name: {}.".format(optimizer_name))
        self.optimizer = self.optimizer_dict[optimizer_name](self.parameters(), lr=self.learning_rate)

        self.gpu = gpu

        self.device = torch.device('cuda'.format(self.gpu) if torch.cuda.is_available() else 'cpu')
        print("Device: {}.".format(self.device))

    def forward(self, x):
        embed = self.embedding(x)
        avg_embed = torch.mean(embed, dim=1)
        hidden = self.fc1(avg_embed)
        hidden = self.dropout(hidden)
        out = self.fc_out(hidden)
        return out

    def train_model(self, model, data_generator, input_path, output_path, word_dict,
                    input_path_val=None, output_path_val=None,
                    input_path_test=None, output_path_test=None,
                    save_folder=None):
        model.to(self.device)
        best_score = 1000
        for epoch in range(self.num_epochs):
            total_y, total_pred = [], []
            total_loss = 0
            step_num = 0
            sample_num = 0
            model.train()
            count = 0
            for x, y in data_generator(input_path, output_path, word_dict, batch_size=self.batch_size):
                time1 = time.time()
                batch_x = torch.LongTensor(x).to(self.device)
                batch_y = torch.FloatTensor(y).to(self.device)
                batch_pred = model(batch_x)
                model.zero_grad()
                loss = self.criterion(batch_pred, batch_y)
                loss.backward()
                self.optimizer.step()

                pred = list(batch_pred.cpu().detach().numpy())
                total_y += list(y)
                total_pred += pred

                total_loss += loss.item() * len(y)
                step_num += 1
                sample_num += len(y)

                time2 = time.time()

                if count % 1000 == 0:
                    print("batch {} time {} s".format(count, time2 - time1))
                count = count + 1

            print("Have trained {} steps.".format(step_num))
            metric = cal_all
            metric_score = metric(np.array(total_y), np.array(total_pred))
            sorted_metric_score = sorted(metric_score.items(), key=lambda x: x[0])
            metrics_string = '\t'.join(['loss'] + [metric_name for metric_name, _ in sorted_metric_score])
            score_string = '\t'.join(
                ['{:.2f}'.format(total_loss / sample_num)] + ['{:.2f}'.format(score) for _, score in
                                                              sorted_metric_score])
            print("{}\t{}\t{}".format('train', epoch, metrics_string))
            print("{}\t{}\t{}".format('train', epoch, score_string))

            if input_path_val and output_path_val:
                metric_score = self.eval_model(model, data_generator, input_path_val, output_path_val, word_dict, 'val',
                                               epoch)
                mse = metric_score['mse']
                torch.save(model, '{}{}.ckpt'.format(save_folder, epoch))
                print("Save model to {}.".format('{}{}.ckpt'.format(save_folder, epoch)))
                if mse < best_score:
                    best_score = mse
                    torch.save(model, '{}{}.ckpt'.format(save_folder, 'best'))
                    print("Save model to {}.".format('{}{}.ckpt'.format(save_folder, 'best')))

            if input_path_test and output_path_test:
                self.eval_model(model, data_generator, input_path_test, output_path_test, word_dict, 'test', epoch)

        if input_path_test and output_path_test:
            model = torch.load('{}{}.ckpt'.format(save_folder, 'best'))
            print(model)
            model.eval()
            self.eval_model(model, data_generator, input_path, output_path, word_dict, 'train', 'final')
            self.eval_model(model, data_generator, input_path_test, output_path_test, word_dict, 'test', 'final')
            self.eval_model(model, data_generator, input_path_val, output_path_val, word_dict, 'val', 'final')

    def eval_model(self, model, data_generator, input_path, output_path, word_dict, phase, epoch):
        model.to(self.device)
        model.eval()
        total_y, total_pred = [], []
        total_loss = 0
        step_num = 0
        sample_num = 0
        for x, y in data_generator(input_path, output_path, word_dict, batch_size=self.batch_size):
            batch_x = torch.LongTensor(x).to(self.device)
            batch_y = torch.FloatTensor(y).to(self.device)
            batch_pred = model(batch_x)
            loss = self.criterion(batch_pred, batch_y)

            pred = list(batch_pred.cpu().detach().numpy())
            total_y += list(y)
            total_pred += pred

            total_loss += loss.item() * len(y)
            step_num += 1
            sample_num += len(y)

        print("Have {} {} steps.".format(phase, step_num))
        metric = cal_all
        metric_score = metric(np.array(total_y), np.array(total_pred))
        sorted_metric_score = sorted(metric_score.items(), key=lambda x: x[0])
        metrics_string = '\t'.join(['loss'] + [metric_name for metric_name, _ in sorted_metric_score])
        score_string = '\t'.join(
            ['{:.2f}'.format(total_loss / sample_num)] + ['{:.2f}'.format(score) for _, score in sorted_metric_score])
        print("{}\t{}\t{}".format(phase, epoch, metrics_string))
        print("{}\t{}\t{}".format(phase, epoch, score_string))
        return metric_score


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')
